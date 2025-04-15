#include <cuda_runtime.h>

#define NUM_PER_THREAD_REDUCE 4
#define WARP_SIZE 32


inline __device__ float my_pow(float a, double b) {
  return pow(a, static_cast<float>(b));
}

inline __device__ void WarpReduce(float *x1, float *x2) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    x1[0] += __shfl_down_sync(0xffffffff, x1[0], delta);
    x2[0] += __shfl_down_sync(0xffffffff, x2[0], delta);
  }
}

inline __device__ void BlockReduce(const int col, float *x1, float *x2, float *x1_addr, float *x2_addr) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  extern __shared__ float share_mem[];
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 2;
    share_mem[offset] = x1[0];
    share_mem[offset + 1] = x2[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 2;
      share_mem[threadIdx.x * 2] += share_mem[offset];
      share_mem[threadIdx.x * 2 + 1] += share_mem[offset + 1];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    x1_addr[col] = share_mem[0];
    x2_addr[col] = share_mem[1];
  }
  __syncthreads();
}


inline __device__ void DsAndDbThreadReduce(const int col, const int row_dim, const int col_dim,
                                           const float *dy, const float *x, float *dscale, float *dbias) {
  int loop_num = (row_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= row_dim) {
        return;
      }

      int pos = col * row_dim + row;
      dscale[0] += dy[pos] * x[pos];
      dbias[0] += dy[pos];
    }
  }
}

extern "C"
__global__ void CalDsAndDbKernel(const int row_dim, const int col_dim, const float *dy, const float *x,
                                 float *dscale_addr, float *dbias_addr) {
  for (int col = blockIdx.x; col < col_dim; col += gridDim.x) {
    float dscale = 0;
    float dbias = 0;
    DsAndDbThreadReduce(col, row_dim, col_dim, dy, x, &dscale, &dbias);
    WarpReduce(&dscale, &dbias);
    BlockReduce(col, &dscale, &dbias, dscale_addr, dbias_addr);
  }
}

inline __device__ void GammaAndBetaThreadReduce(const int col, const int batch, const int num_channel,
                                                const int num_groups, const float *dscale, const float *dbias,
                                                const float *mean, const float *rstd, float *dg, float *db) {
  int loop_num = (batch + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= batch) {
        return;
      }

      int idx1 = row * num_channel + col;
      int idx2 = idx1 * num_groups / num_channel;
      dg[0] += (dscale[idx1] - dbias[idx1] * mean[idx2]) * rstd[idx2];
      db[0] += dbias[idx1];
    }
  }
}

extern "C"
__global__ void GammaAndBetaPropKernel(const int batch, const int num_channel, const int num_groups,
                                       const float *dscale, const float *dbias, const float *mean_addr,
                                       const float *rstd_addr, float *dg_addr, float *db_addr) {
  for (int col = blockIdx.x; col < num_channel; col += gridDim.x) {
    float dg = 0;
    float db = 0;
    GammaAndBetaThreadReduce(col, batch, num_channel, num_groups, dscale, dbias, mean_addr, rstd_addr, &dg, &db);
    WarpReduce(&dg, &db);
    BlockReduce(col, &dg, &db, dg_addr, db_addr);
  }
}

inline __device__ void InputThreadReduce(const int row, const int col_dim, const int num_channel, const int HxW,
                                         float *sum1, float *sum2, float *sum3, const float *dy, const float *x,
                                         const float *mean, const float *rstd, const float *gamma) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        sum1[0] = -0.5 * sum1[0] * my_pow(static_cast<float>(rstd[row]), 3.0);
        sum3[0] = -2.0 * sum3[0];
        return;
      }

      int pos = row * col_dim + col;
      int gamma_offset = (pos / HxW) % num_channel;
      float v1 = dy[pos] * gamma[gamma_offset];
      float v2 = x[pos] - mean[row];

      sum1[0] += v1 * v2;
      sum2[0] += v1;
      sum3[0] += v2;
    }
  }
  sum1[0] = -0.5 * sum1[0] * my_pow(static_cast<float>(rstd[row]), 3.0);
  sum3[0] = -2.0 * sum3[0];
}

inline __device__ void InputWarpReduce(float *sum1, float *sum2, float *sum3) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    sum1[0] += __shfl_down_sync(0xffffffff, sum1[0], delta);
    sum2[0] += __shfl_down_sync(0xffffffff, sum2[0], delta);
    sum3[0] += __shfl_down_sync(0xffffffff, sum3[0], delta);
  }
}

inline __device__ void InputBlockReduce(const int col_dim, float *sum1, float *sum2, float *sum3, float *share_mem) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = sum1[0];
    share_mem[offset + 1] = sum2[0];
    share_mem[offset + 2] = sum3[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 3;
      share_mem[threadIdx.x * 3] += share_mem[offset];
      share_mem[threadIdx.x * 3 + 1] += share_mem[offset + 1];
      share_mem[threadIdx.x * 3 + 2] += share_mem[offset + 2];
    }
  }
  __syncthreads();
}

inline __device__ void InputProp(const int row, const int col_dim, const int num_channel, const int HxW, const float *dy,
                                 const float *x, const float *mean, const float *rstd, const float *gamma, float *dx,
                                 const float *share_mem) {
  float v3 = rstd[row];
  float v4 = share_mem[0] * (2.0 / col_dim);
  float v5 = (-1.0 * v3 * share_mem[1] + (1.0 / col_dim) * share_mem[0] * share_mem[2]) * (1.0 / col_dim);
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = (pos / HxW) % num_channel;
    float v1 = dy[pos] * gamma[gamma_offset];
    float v2 = x[pos] - mean[row];
    dx[pos] = (float)(v1 * v3 + v4 * v2 + v5);
  }
}

extern "C"
__global__ void InputPropKernel(const int row_dim, const int col_dim, const int num_channel, const int HxW, const float *dy,
                                const float *x, const float *mean, const float *rstd, const float *gamma, float *dx) {
  for (int row = blockIdx.x; row < row_dim; row += gridDim.x) {
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
   
    extern __shared__ float share_mem[];
    InputThreadReduce(row, col_dim, num_channel, HxW, &sum1, &sum2, &sum3, dy, x, mean, rstd, gamma);
    //printf("a1dy:%f",dy[0]);
    InputWarpReduce(&sum1, &sum2, &sum3);
    //printf("a2dy:%f",dy[0]);
    InputBlockReduce(col_dim, &sum1, &sum2, &sum3, share_mem);
    //printf("a3dy:%f",dy[0]);
    InputProp(row, col_dim, num_channel, HxW, dy, x, mean, rstd, gamma, dx, share_mem);
 
  }
}


