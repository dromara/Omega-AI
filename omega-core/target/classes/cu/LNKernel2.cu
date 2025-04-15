#include "cuda_fp16.h"

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

__device__ void cuChanOnlineSum(
  const float muB,
  const float sigma2B,
  const float countB,
  float& mu,
  float& sigma2,
  float& count)
{
  float delta = muB - mu;
  float nA = count;
  float nB = countB;
  count = count + countB;
  float nX = count;
  if (nX > 0.0f) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = 0.0f;
    sigma2 = 0.0f;
  }
}

__device__ void cuWelfordMuSigma2(
  const float* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = 0.0f;
  sigma2 = 0.0f;
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const float* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr,mu,sigma2,count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float curr = lvals[l+k];
      	cuWelfordOnlineSum(curr,mu,sigma2,count);
      }
    }
    for (;  l < n2;  ++l) {
      float curr = lvals[l];
      cuWelfordOnlineSum(curr,mu,sigma2,count);
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      float sigma2B = __shfl(sigma2, srcLaneB);
      float muB = __shfl(mu, srcLaneB);
      float countB = __shfl(count, srcLaneB);
      cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y+1] = sigma2;
          ubuf[2*wrt_y] = mu;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2*threadIdx.y+1];
          float muB = ubuf[2*threadIdx.y];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
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
      sigma2 = ubuf[1]/n2;
      // don't care about final value of count, we know count == n2
    } else {
      mu = __shfl(mu, 0);
      sigma2 = __shfl(sigma2/n2, 0);
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
    extern __shared__ float buf[];
    float mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf);

    const float* lvals = vals + i1*n2;
    float* ovals = output_vals + i1*n2;
    float c_invvar = rsqrtf(sigma2 + epsilon);
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