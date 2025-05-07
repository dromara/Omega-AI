
#include <cuda_runtime.h>

constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;

inline __device__ void MeanAndVarAccumulation(float *mean, float *var, float *count, const float &val) {
  // Welford Algorithm:
  // \mu_k = \mu_{k-1} + (x_k - \mu_{k-1})/k
  // \sigma_k^2 = \sigma_{k-1}^2 + (x_k - \mu_{k-1}) * (x_k - \mu_k)
  count[0]++;
  float mean_new = mean[0] + (val - mean[0]) / count[0];
  var[0] = var[0] + (val - mean[0]) * (val - mean_new);
  mean[0] = mean_new;
}

inline __device__ void MeanAndVarMerge(float *mean1, float *var1, float *count1, const float &mean2, const float &var2, const float &count2) {
  float zero = 0;
  if (count2 == zero) {
    return;
  }

  float count = count1[0] + count2;
  var1[0] = var1[0] + var2 + (mean1[0] - mean2) * (mean1[0] - mean2) * count1[0] * count2 / count;
  mean1[0] = (count1[0] * mean1[0] + count2 * mean2) / count;
  count1[0] = count;
}

inline __device__ void ThreadReduce(const int col_dim, const float *block_addr, float *mean, float *var, float *count) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int pos = NUM_PER_THREAD_REDUCE * i + j;
      if (pos >= col_dim) {
        return;
      }
      MeanAndVarAccumulation(mean, var, count, static_cast<float>(block_addr[pos]));
    }
  }
}

inline __device__ void WarpReduce(float *mean, float *var, float *count) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    float mean_other = __shfl_down_sync(0xffffffff, mean[0], delta);
    float var_other = __shfl_down_sync(0xffffffff, var[0], delta);
    float count_other = __shfl_down_sync(0xffffffff, count[0], delta);
    MeanAndVarMerge(mean, var, count, mean_other, var_other, count_other);
  }
}

inline __device__ void BlockReduce(const int col_dim, float *mean, float *var, float *count, float *mean_addr,
                                   float *rstd_addr, float *share_mem, const float epsilon) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = mean[0];
    share_mem[offset + 1] = var[0];
    share_mem[offset + 2] = count[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 3;
      MeanAndVarMerge(&share_mem[threadIdx.x * 3], &share_mem[threadIdx.x * 3 + 1], &share_mem[threadIdx.x * 3 + 2],
                      share_mem[offset], share_mem[offset + 1], share_mem[offset + 2]);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mean_addr[blockIdx.x] = static_cast<float>(share_mem[0]);
    share_mem[1] = 1.0 / sqrtf((share_mem[1] / col_dim + epsilon));
    rstd_addr[blockIdx.x] = static_cast<float>(share_mem[1]);
  }
}

inline __device__ void GroupNorm(const int row, const int col_dim, const int num_channel, const int HxW, const float *x,
                                 const float *share_mem, const float *gamma, const float *beta, float *y) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = row * col_dim + col;
    int i = (pos / HxW) % num_channel;
    float tmp_y = (static_cast<float>(x[pos]) - share_mem[0]) * share_mem[1] *
                   static_cast<float>(gamma[i]) + static_cast<float>(beta[i]);
    y[pos] = (float)(tmp_y);
  }
}



extern "C"
__global__ void GroupNormKernel(const int row_dim, const int col_dim, const int num_channel, const int HxW,
                                const float epsilon, const float *x, const float *gamma, const float *beta, float *y,
                                float *mean_addr, float *rstd_addr) {
  for (auto row = blockIdx.x; row < row_dim; row += gridDim.x) {
    float mean = 0;
    float var = 0;
    float count = 0;
    const float *block_addr = x + row * col_dim;
    extern __shared__ float share_mem[];

    ThreadReduce(col_dim, block_addr, &mean, &var, &count);
    WarpReduce(&mean, &var, &count);
    BlockReduce(col_dim, &mean, &var, &count, mean_addr, rstd_addr, share_mem, epsilon);

    __syncthreads();
    GroupNorm(row, col_dim, num_channel, HxW, x, share_mem, gamma, beta, y);
  }
}
