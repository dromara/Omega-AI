#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;

extern "C"
__device__ __forceinline__ float WARP_SHFL_DOWN(float value, unsigned int delta,int width = warpSize, unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

extern "C"
__inline__ __device__ float WarpReduceSum(float val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

__inline__ __device__ float BlockReduceSum(float val, float *shared) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / C10_WARP_SIZE)) ? shared[lid] : 0.0f;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

extern "C"
__inline__ __device__ float BlockReduceSum(float val, float *shared, int max_num_threads) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (min(blockDim.x, max_num_threads) / C10_WARP_SIZE))
            ? shared[lid]
            : 0.0f;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

extern "C"
__global__ void LayerNormFusedForwardKernel(int N,
                                            float eps,
                                            float const *X,
                                            float *mean,
                                            float *rstd,
                                            float const *gamma,
                                            float const *beta,
                                            float *Y) {
  __shared__ float m_shared[C10_WARP_SIZE];
  __shared__ float v_shared[C10_WARP_SIZE];
  const int i = blockIdx.x;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = threadIdx.x; j < N; j += min(blockDim.x, kCUDABlockReduceNumThreads)) {
    const int64_t index = i * N + j;
    sum1 += X[index];
    sum2 += X[index] * X[index];
  }
  if (threadIdx.x < kCUDABlockReduceNumThreads) {
    sum1 = BlockReduceSum(sum1, m_shared, min(blockDim.x, kCUDABlockReduceNumThreads));
    sum2 = BlockReduceSum(sum2, v_shared, min(blockDim.x, kCUDABlockReduceNumThreads));
  }
  if (threadIdx.x == 0) {
    float const scale = 1.0f / N;
    sum1 *= scale;
    sum2 = max(sum2 * scale - sum1 * sum1, 0.0f);
    mean[i] = sum1;
    rstd[i] = rsqrt(sum2 + eps);
  }

  __syncthreads();

  for (int64_t j = threadIdx.x; j < N; j += min(blockDim.x, kCUDANumThreads)) {
    const int64_t index = i * N + j;
    const float gamma_v = gamma == nullptr ? 1.0f : gamma[j];
    const float beta_v = beta == nullptr ? 0.0f : beta[j];
    Y[index] = (X[index] - mean[i]) * rstd[i] * gamma_v + beta_v;
  }
}

extern "C"
__global__ void ComputeInternalGradientsCUDAKernel(int N, float const *dY, float const *X, float const *gamma, float *ds, float *db) {
  __shared__ float ds_shared[C10_WARP_SIZE];
  __shared__ float db_shared[C10_WARP_SIZE];
  const int i = blockIdx.x;
  float sum1 = 0;
  float sum2 = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
    const float gamma_v = gamma == nullptr ? 1.0f : gamma[j];
    sum1 += dY[index] * X[index] * gamma_v;
    sum2 += dY[index] * gamma_v;
  }
  sum1 = BlockReduceSum(sum1, ds_shared);
  sum2 = BlockReduceSum(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
}

extern "C"
__global__ void ComputeGradientFusedParamsCUDAKernel(int M,int N,float const *mean,float const *rstd,float const *ds,float const *db,float *c1,float *c2) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < M) {
    const float s = 1.0f / N;
    const float a = (db[index] * mean[index] - ds[index]) * rstd[index] * rstd[index] * rstd[index] * s;
    c1[index] = a;
    c2[index] = -(a * mean[index] + db[index] * rstd[index] * s);
  }
}

__device__ __inline__ void compute_gI(float const *__restrict__ dY,
                                      float const *__restrict__ X,
                                      float const *__restrict__ mean,
                                      float const *__restrict__ rstd,
                                      float const *__restrict__ gamma,
                                      float *dX,
                                      int const N,
                                      float *buf) {
  int const i1 = blockIdx.x;
  const float mean_val = mean[i1];
  const float rstd_val = rstd[i1];
  float stats_x1{0}, stats_x2{0};
  constexpr int unroll = 4;
  int l = unroll * threadIdx.x;
  float const *X_i = X + i1 * N;
  float const *dY_i = dY + i1 * N;
  float *dX_i = dX + i1 * N;
  // vectorized reads don't improve perf, so use regular unrolling

  for (; l + unroll - 1 < N; l += blockDim.x * unroll) {
#pragma unroll
    for (int k = 0; k < unroll; k++) {
      float gamma_val = (gamma != nullptr) ? gamma[l + k] : 1.0f;
      const float c_h = X_i[l + k];
      const float c_loss = dY_i[l + k];
      stats_x1 += c_loss * gamma_val;
      stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }
  }
  for (; l < N; l++) {
    float gamma_val = (gamma != nullptr) ? gamma[l] : 1.0f;
    const float c_h = X_i[l];
    const float c_loss = dY_i[l];
    stats_x1 += c_loss * gamma_val;
    stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
  }

  stats_x1 = BlockReduceSum(stats_x1, buf);
  stats_x2 = BlockReduceSum(stats_x2, buf);
  if (threadIdx.x == 0) {
    buf[0] = stats_x1;
    buf[1] = stats_x2;
  }
  __syncthreads();
  stats_x1 = buf[0];
  stats_x2 = buf[1];
  float fH = N;
  float term1 = (1.0f / fH) * rstd_val;

  for (int l = threadIdx.x; l < N; l += blockDim.x) {
    const float x = X_i[l];
    const float dy = dY_i[l];
    float gamma_val = (gamma != nullptr) ? gamma[l] : 1.0f;
    float f_grad_input = fH * gamma_val * dy;
    f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
    f_grad_input -= stats_x1;
    f_grad_input *= term1;
    dX_i[l] = f_grad_input;
  }
}


extern "C"
__global__ void layer_norm_grad_input_kernel(float const *__restrict__ dY,
                                             float const *__restrict__ X,
                                             float const *__restrict__ mean,
                                             float const *__restrict__ rstd,
                                             float const *__restrict__ gamma,
                                             float *dX,
                                             int const N) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  float *buf = reinterpret_cast<float *>(&s_data1);

  compute_gI(dY, X, mean, rstd, gamma, dX, N, buf);
}

extern "C"
__global__ void GammaBetaBackwardSimpleCUDAKernel(int M,
                                                  int N,
                                                  float const *dY,
                                                  float const *X,
                                                  float const *mean,
                                                  float const *rstd,
                                                  float *dg,
                                                  float *db) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    float sum1 = 0;
    float sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dg == nullptr ? 0.0f : dY[index] * (X[index] - mean[i]) * rstd[i];
      sum2 += db == nullptr ? 0.0f : dY[index];
    }
    if (dg != nullptr) {
      dg[j] = sum1;
    }
    if (db != nullptr) {
      db[j] = sum2;
    }
  }
}

extern "C"
__global__ void GammaBetaBackwardCUDAKernel(int M,
                                            int N,
                                            float const *dY,
                                            float const *X,
                                            float const *mean,
                                            float const *rstd,
                                            float *dg,
                                            float *db) {
  __shared__ float g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ float b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  float dg_sum1 = 0;
  float dg_sum2 = 0;
  float db_sum1 = 0;
  float db_sum2 = 0;
  if (j < N) {
    for (int64_t i = threadIdx.y; i < M; i += blockDim.y * 2) {
      const int i1 = i;
      const int i2 = i + blockDim.y;
      const int index1 = i1 * N + j;
      const int index2 = i2 * N + j;
      dg_sum1 += dg == nullptr ? 0.0f : dY[index1] * (X[index1] - mean[i1]) * rstd[i1];
      db_sum1 += db == nullptr ? 0.0f : dY[index1];
      if (i2 < M) {
        dg_sum2 += dg == nullptr ? 0.0f : dY[index2] * (X[index2] - mean[i2]) * rstd[i2];
        db_sum2 += db == nullptr ? 0.0f : dY[index2];
      }
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();
  float sum1 = g_shared[threadIdx.x][threadIdx.y];
  float sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
}

