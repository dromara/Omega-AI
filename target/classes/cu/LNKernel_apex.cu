#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDA_FP16.h"

__device__ __forceinline__ float WARP_SHFL(float value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

__device__ __forceinline__ float WARP_SHFL_XOR(float value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

__device__ void cuChanOnlineSum(const float muB, const float sigma2B, const float countB,
                                float &mu, float &sigma2, float &count) {
  float delta = muB - mu;
  float nA = count;
  float nB = countB;
  count = count + countB;
  float nX = count;
  if (nX > float(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = float(0);
    sigma2 = float(0);
  }
}

__device__ void cuWelfordOnlineSum(
  const float curr,
  float& mu,
  float& sigma2,
  float& count)
{
  count = count + 1.0f;
  float delta = curr - mu;
  float lmean = mu + delta / count;
  mu = lmean;
  float delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

__device__ void cuWelfordMuSigma2(const float *__restrict__ vals, const int n1,
                                  const int n2, const int i1, float &mu, float &sigma2,
                                  float *buf) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = float(0);
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const float *lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        float curr = static_cast<float>(lvals[l + k]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float muB = WARP_SHFL(mu, srcLaneB);
      float countB = WARP_SHFL(count, srcLaneB);
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float *ubuf = (float *)buf;
      float *ibuf = (float *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);
    }
  }
}

__device__ void cuApplyLayerNorm_(
  float* __restrict__ output_vals,
  float* __restrict__ mean,
  float* __restrict__ invvar,
  const float* __restrict__ vals,
  const int n1,
  const int n2,
  const float epsilon,
  const float* __restrict__ gamma,
  const float* __restrict__ beta
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (int i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    extern __shared__ float s_float[];
    float* buf = s_float;
    float mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf);

    const float* lvals = vals + i1*n2;
    float* ovals = output_vals + i1*n2;
    float c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;

    for (int i = thrx;  i < n2;  i+=numx) {
      float curr = lvals[i];
      ovals[i] = gamma[i] * c_invvar * (curr - mu) + beta[i];
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

extern "C"
__global__ void cuApplyLayerNorm(
  float* __restrict__ output_vals,
  float* __restrict__ mean,
  float* __restrict__ invvar,
  const float* __restrict__ vals,
  const int n1,
  const int n2,
  const float epsilon,
  const float* __restrict__ gamma,
  const float* __restrict__ beta
  )
{
  cuApplyLayerNorm_(output_vals, mean, invvar, vals, n1, n2, epsilon, gamma, beta);
}

__device__ float clamp_by_magnitude(float curr_gamma, float eps)
{
  const float kMinGamma = float(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}

extern "C" 
__global__ void cuComputeGradInput(
    const float* __restrict__ dout,
    const float* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    float epsilon,
    const float* gamma,
    const float* beta,
    float* grad_input,
    const float eps)
{
  for (int i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);
    const float* k_h = input_or_output + i1*n2;
    const float* k_dout = dout + i1*n2;
    const float c_invvar = invvar[i1];
    const float c_mean = mean[i1];
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const float c_h = static_cast<float>(k_h[l+k]);
          const float c_loss = static_cast<float>(k_dout[l+k]);
          sum_loss1 += c_loss * gamma[l+k];
          sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const float c_h = static_cast<float>(k_h[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        sum_loss1 += c_loss * gamma[l];
        sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const float c_h = static_cast<float>(k_h[l+k]);
          const float c_loss = static_cast<float>(k_dout[l+k]);
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const float c_h = static_cast<float>(k_h[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      extern __shared__ float s_float[];
      float* buf = s_float;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const float c_h = static_cast<float>(k_h[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        const float k_gamma = static_cast<float>(clamp_by_magnitude(gamma[l], eps));
        float f_grad_input = fH * c_loss * k_gamma;
        const float k_beta = beta[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<float>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const float c_h = static_cast<float>(k_h[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        float f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<float>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

extern "C" 
__global__ void cuComputeGradInput2(const float *__restrict__ dout,
                   const float *__restrict__ input, const int n1, const int n2,
                   const float *__restrict__ mean, const float *__restrict__ invvar,
                   float epsilon, const float *gamma, float *grad_input) {
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);
    const float c_mean = mean[i1];
    const float c_invvar = invvar[i1];
    const float *k_input = input + i1 * n2;
    const float *k_dout = dout + i1 * n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const float c_h = static_cast<float>(k_input[l + k]);
          const float c_loss = static_cast<float>(k_dout[l + k]);
          sum_loss1 += c_loss * static_cast<float>(gamma[l + k]);
          sum_loss2 += c_loss * static_cast<float>(gamma[l + k]) * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const float c_h = static_cast<float>(k_input[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        sum_loss1 += c_loss * static_cast<float>(gamma[l]);
        sum_loss2 += c_loss * static_cast<float>(gamma[l]) * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const float c_h = static_cast<float>(k_input[l + k]);
          const float c_loss = static_cast<float>(k_dout[l + k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const float c_h = static_cast<float>(k_input[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      extern __shared__ float s_float[];
      float* buf = s_float;
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const float c_h = static_cast<float>(k_input[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        float f_grad_input = fH * c_loss * static_cast<float>(gamma[l]);
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<float>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const float c_h = static_cast<float>(k_input[l]);
        const float c_loss = static_cast<float>(k_dout[l]);
        float f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<float>(f_grad_input);
      }
    }
  }
}
