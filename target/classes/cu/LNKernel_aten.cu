
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

#define CAFFE_CUDA_NUM_THREADS 128

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T>
inline __host__ __device__ T Cube(const T x) {
  return x * x * x;
}

extern "C"
__global__ void ComputeSigmaAndFusedParamsCUDAKernel(                
      const int N,                                                        
      const float eps,                                                        
      const float* mean,                                                      
      const float* var,                                                       
      float* sigma,                                                           
      float* scale,                                                           
      float* bias) {                                                          
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;  
    if (index < N) {                                                      
      const float rstd = rsqrtf(var[index] + eps);                         
      sigma[index] = rstd * (var[index] + eps);                           
      scale[index] = rstd;                                                
      bias[index] = -rstd * mean[index];                                  
    }                                                                     
}

extern "C"
__global__ void LayerNormForwardCUDAKernel(
    const int M,
    const int N,
    const float* X,
    const float* scale,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* Y) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    const int j = index % N;
    Y[index] = (X[index] * scale[i] + bias[i]) * gamma[j] + beta[j];
  }
}

extern "C"
__global__ void RowwiseMomentsCUDAKernel(const int cols, const float* X, float* mean, float* var) {
  __shared__ typename BlockReduce<float>::TempStorage m_storage;
  __shared__ typename BlockReduce<float>::TempStorage v_storage;
  const float scale = float(1) / static_cast<float>(cols);
  const int r = blockIdx.x;
  float m_val = 0;
  float v_val = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const int X_index = r * cols + c;
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    m_val += __ldg(X + X_index);
    v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
    m_val += X[X_index];
    v_val += X[X_index] * X[X_index];
#endif
  }
  m_val = BlockReduce<float>(m_storage).Sum(m_val);
  v_val = BlockReduce<float>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    const float mu = m_val * scale;
    mean[r] = mu;
    var[r] = v_val * scale - mu * mu;
  }
}

extern "C"
__global__ void ComputeInternalGradientsCUDAKernel(
    const int N,
    const float *const dYxX,
    const float *const dY,
    const float *const gamma,
    float *const ds,
    float *const db) {
  __shared__ typename BlockReduce<float>::TempStorage ds_storage;
  __shared__ typename BlockReduce<float>::TempStorage db_storage;
  const int i = blockIdx.x;
  float ds_val = 0;
  float db_val = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
    ds_val += __ldg(dYxX + index) * __ldg(gamma + j);
    db_val += __ldg(dY + index) * __ldg(gamma + j);
#else
    ds_val += dYxX[index] * gamma[j];
    db_val += dY[index] * gamma[j];
#endif
  }
  ds_val = BlockReduce<float>(ds_storage).Sum(ds_val);
  db_val = BlockReduce<float>(db_storage).Sum(db_val);
  if (threadIdx.x == 0) {
    ds[i] = ds_val;
    db[i] = db_val;
  }
}

extern "C"
__global__ void ComputeFusedParamsCUDAKernel(
    const int M,
    const int N,
    const float* mean,
    const float* sigma,
    const float* ds,
    const float* db,
    float* rstd,
    float* X_scale,
    float* bias,
    float* g_scale) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M) {
    const float scale = float(1) / static_cast<float>(N);
    const float rstd_val = float(1) / sigma[index];
    const float X_scale_val = (db[index] * mean[index] - ds[index]) * Cube<float>(rstd_val) * scale;
    rstd[index] = rstd_val;
    X_scale[index] = X_scale_val;
    bias[index] = -(X_scale_val * mean[index] + db[index] * rstd_val * scale);
    if (g_scale != nullptr) {
      g_scale[index] = -rstd_val * mean[index];
    }
  }
}

extern "C"
__global__ void LayerNormBackwardCUDAKernel(
    const int M,
    const int N,
    const float* dY,
    const float* X,
    const float* gamma,
    const float* dY_scale,
    const float* X_scale,
    const float* bias,
    float* dX) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    const int j = index % N;
    dX[index] = dY[index] * dY_scale[i] * gamma[j] + X[index] * X_scale[i] + bias[i];
  }
}