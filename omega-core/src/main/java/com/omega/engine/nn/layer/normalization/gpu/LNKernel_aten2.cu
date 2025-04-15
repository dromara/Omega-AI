
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDA_FP16.h"

#define C10_WARP_SIZE 32

constexpr int vec_size = 4;

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

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  __inline__ __device__ WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  __inline__ __device__ WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  U new_mean = curr_sum.mean + delta * (1.f/new_count); //proper division is slow, this is less accurate but noticeably faster
  return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

__device__
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA
) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean, sigma2;
  if (count > decltype(dataB.count){0}) {
    auto coef = 1.f/count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA*dataA.mean + nB*dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  } else {
    mean = U(0);
    sigma2 = U(0);
  }
  return {mean, sigma2, count};
}

template<typename T>
__device__ WelfordDataLN compute_stats(
  const T*  __restrict__ X,
  const int N,
  float * buf
  ) {
    //X points to the row to read
    using vec_t = aligned_vector<T, vec_size>;

    const vec_t * X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;
    WelfordDataLN wd(0.f, 0.f, 0.f);
    //no tail, we check that N is multiple of vec_size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++){
        wd = cuWelfordOnlineSum(static_cast<float>(data.val[ii]), wd);
      }
    }
    // intra-warp reduction
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + blockDim.y;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          WelfordDataLN wdB{meansigmabuf[2*threadIdx.y],
                          meansigmabuf[2*threadIdx.y+1],
                          countbuf[threadIdx.y]};
          wd = cuWelfordCombine(wd, wdB);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0 && threadIdx.y ==0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      __syncthreads();
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1],0.f};

    } else {
      return WelfordDataLN{WARP_SHFL(wd.mean,0), WARP_SHFL(wd.sigma2,0)/float(N), 0.f};
    }
}

template <typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    extern __shared__ float s_data[]; //if we made smem WelfordDataLN type, there would be bank conflicts,
    //as one thread would have to write 3 consecutive floats
    auto i1 = blockIdx.x;
    const T * block_row = X + i1 * N;
    WelfordDataLN wd = compute_stats(block_row, N, s_data);

    using vec_t = aligned_vector<T, vec_size>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t * gamma_vec = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    const vec_t * beta_vec = (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;

    T_ACC rstd_val = rsqrt(wd.sigma2 + eps);

    // No tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      // Computation is performed in T_ACC, X is cast to T_ACC and result is implicitly cast to T
      if (gamma_vec != nullptr && beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
            + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else if (gamma_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        }
      }
      Y_vec[i] = out;
    }
    if (thrx == 0) {
      mean[i1] = wd.mean;
      rstd[i1] = rstd_val;
    }
}

template <typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int /*N*/,
  T_ACC /*eps*/,
  const  T* __restrict__ /*X*/,
  const  T* /*gamma*/,
  const  T* /*beta*/,
  T_ACC* /*mean*/,
  T_ACC* /*rstd*/,
  T* /*Y*/){
    CUDA_KERNEL_ASSERT(false && "doesn't work with double");
  }
  

extern "C"
__global__ void vectorized_layer_norm_kernel(
  const int N,
  float eps,
  const  float* __restrict__ X,
  const  float* gamma,
  const  float* beta,
  float* mean,
  float* rstd,
  float* Y){
    vectorized_layer_norm_kernel_impl(N, eps, X, gamma, beta, mean, rstd, Y);
  }

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / C10_WARP_SIZE;
    }
};

template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

extern "C" 
__global__ void layer_norm_grad_input_kernel_vectorized(
  const float* __restrict__ dY,
  const float* __restrict__ X,
  const float* __restrict__ mean,
  const float* __restrict__ rstd,
  const float* __restrict__ gamma,
  float* dX,
  const int N) {
  alignas(sizeof(double)) extern __shared__ char shared_data[];
  float* reduce_buf = reinterpret_cast<float*>(&shared_data);

  const auto bIdx = blockIdx.x;
  const float mean_val = mean[bIdx];
  const float rstd_val = rstd[bIdx];
  const float* X_i = X + bIdx * N;
  const float* dY_i = dY + bIdx * N;
  float* dX_i = dX + bIdx * N;

  using vec_t = aligned_vector<float, vec_size>;
  const vec_t* const X_i_vec_ptr = reinterpret_cast<const vec_t*>(X_i);
  const vec_t* const dY_i_vec_ptr = reinterpret_cast<const vec_t*>(dY_i);
  const vec_t* const gamma_vec_ptr = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
  vec_t* const dX_i_vec = reinterpret_cast<vec_t*>(dX_i);

  vec_t X_i_vec_reg, dY_i_vec_reg, gamma_vec_reg, dX_i_vec_reg;
  for (int k = 0; k < vec_size; ++k) {
    gamma_vec_reg.val[k] = float(1);
  }

  float stats_x1{0}, stats_x2{0};
  unsigned int l = threadIdx.x * vec_size;
  for (; l + vec_size - 1 < N; l += blockDim.x * vec_size) {
    unsigned int vec_idx = l / vec_size;
    if (gamma != nullptr) {
      gamma_vec_reg = gamma_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < vec_size; ++k) {
      const auto gamma_val = static_cast<float>(gamma_vec_reg.val[k]);
      const auto c_h = static_cast<float>(X_i_vec_reg.val[k]);
      const auto c_loss = static_cast<float>(dY_i_vec_reg.val[k]);
      stats_x1 += c_loss * gamma_val;
      stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }
  }

  // Tail Loop
  for (; l < N; l++) {
    const auto gamma_val = (gamma != nullptr) ? static_cast<float>(gamma[l]) : float(1);
    const auto c_h = static_cast<float>(X_i[l]);
    const auto c_loss = static_cast<float>(dY_i[l]);
    stats_x1 += c_loss * gamma_val;
    stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
  }

  // Reduction in Shared Memory
  stats_x1 = BlockReduceSum(stats_x1, reduce_buf);
  stats_x2 = BlockReduceSum(stats_x2, reduce_buf);
  if (threadIdx.x == 0) {
    reduce_buf[0] = stats_x1;
    reduce_buf[1] = stats_x2;
  }
  __syncthreads();
  stats_x1 = reduce_buf[0];
  stats_x2 = reduce_buf[1];

  float fH = N;
  float term1 = (float(1) / fH) * rstd_val;

  l = threadIdx.x * vec_size;
  for (; l + vec_size - 1 < N; l += blockDim.x * vec_size) {
    unsigned int vec_idx = l / vec_size;
    if (gamma != nullptr) {
      gamma_vec_reg = gamma_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < vec_size; ++k) {
      const auto gamma_val = static_cast<float>(gamma_vec_reg.val[k]);
      const auto x = static_cast<float>(X_i_vec_reg.val[k]);
      const auto dy = static_cast<float>(dY_i_vec_reg.val[k]);

      float f_grad_input = fH * gamma_val * dy;
      f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
      f_grad_input -= stats_x1;
      f_grad_input *= term1;
      dX_i_vec_reg.val[k] = f_grad_input;
    }

    dX_i_vec[vec_idx] = dX_i_vec_reg;
  }

  // Tail Loop
  for (; l < N; l += blockDim.x) {
    const auto x = X_i[l];
    const auto dy = dY_i[l];
    const auto gamma_val = (gamma != nullptr) ? static_cast<float>(gamma[l]) : float(1);

    float f_grad_input = fH * gamma_val * dy;
    f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
    f_grad_input -= stats_x1;
    f_grad_input *= term1;
    dX_i[l] = f_grad_input;
  }
} 

template<typename T, typename T_ACC>
__device__ __inline__ void compute_gI(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T* dX,
  const int N,
  T_ACC * buf){
    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};
    constexpr int unroll = 4;
    auto l = unroll * threadIdx.x;
    const T * X_i = X + i1 * N;
    const T * dY_i = dY + i1 * N;
    T * dX_i = dX + i1 * N;
    //vectorized reads don't improve perf, so use regular unrolling

    for (; l+unroll - 1 < N; l += blockDim.x * unroll){
      #pragma unroll
      for (int k=0; k< unroll; k++){
          const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l+k]) : T_ACC(1);
          const auto c_h = static_cast<T_ACC>(X_i[l+k]);
          const auto c_loss = static_cast<T_ACC>(dY_i[l+k]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    }
    for (;  l < N; l ++) {
          const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);
          const auto c_h = static_cast<T_ACC>(X_i[l]);
          const auto c_loss = static_cast<T_ACC>(dY_i[l]);
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
    T_ACC fH = N;
    T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

    for (int l = threadIdx.x; l < N; l += blockDim.x){
        const auto x = X_i[l];
        const auto dy = dY_i[l];
        const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);

        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        dX_i[l] = f_grad_input;
    }
}

extern "C"
__global__ void layer_norm_grad_input_kernel(
  const float* __restrict__ dY,
  const float* __restrict__ X,
  const float* __restrict__ mean,
  const float* __restrict__ rstd,
  const float* __restrict__ gamma,
  float*  dX,
  const int N){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    float * buf = reinterpret_cast<float*>(&s_data1);

    compute_gI(dY, X, mean, rstd, gamma, dX, N, buf);
}

extern "C"
__global__ void GammaBetaBackwardCUDAKernel(
    int M,
    int N,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    float* dg,
    float* db) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  float* s_data_typed = reinterpret_cast<float*>(&s_data1);
  float* s_dg;
  float* s_db;

  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  float dg_sum = 0;
  float db_sum = 0;

  if (j < N) {
    constexpr int unroll_factor = 8;

    float mean_reg;
    float rstd_reg;
    float dY_reg;
    float X_reg;

    // Main Loop
    int bcounter;
    for (bcounter = 0; bcounter < M / (blockDim.y * unroll_factor); bcounter++){
      int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;

      #pragma unroll
      for (int ii = 0; ii < unroll_factor; ++ii) {
        dY_reg = dY[(offset + ii) * N + j];
        X_reg = X[(offset + ii) * N + j];
        mean_reg = mean[offset + ii];
        rstd_reg = rstd[offset + ii];
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // Remainder loop
    int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;
    for (int ii = 0; ii < unroll_factor; ii++ ){
      if ((offset + ii) < M) {
        dY_reg = dY[(offset + ii) * N + j ];
        X_reg = X[(offset + ii) * N + j];
        mean_reg = mean[offset + ii];
        rstd_reg = rstd[offset + ii];
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // Do the final reduction in shared memory
    s_dg = s_data_typed;
    s_db = s_data_typed + blockDim.x * blockDim.y;
    s_dg[threadIdx.y * blockDim.x + threadIdx.x] = dg_sum;
    s_db[threadIdx.y * blockDim.x + threadIdx.x] = db_sum;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      if (threadIdx.y < offset) {
        s_dg[threadIdx.y * blockDim.x + threadIdx.x] +=
            s_dg[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
        s_db[threadIdx.y * blockDim.x + threadIdx.x] +=
            s_db[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
        }
      __syncthreads();
    }

    if (threadIdx.y == 0) {
      if (dg) {
        dg[j] = s_dg[threadIdx.x];
      }
      if (db) {
        db[j] = s_db[threadIdx.x];
      }
    }
  }
}


