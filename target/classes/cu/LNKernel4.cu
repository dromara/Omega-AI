
__device__ __forceinline__ float warp_shfl_xor(float value, int laneMask, int width = 32, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

extern "C"
__global__ void LayerNormFusedBackwardKernel_Data(const int nbatch,
                                                  const int nchannel,
                                                  const float* __restrict__ in_data,
                                                  const float* __restrict__ out_grad,
                                                  const float* __restrict__ mean_data,
                                                  const float* __restrict__ std_data,
                                                  const float* __restrict__ gamma,
                                                  float* data_grad,
                                                  const int LOAD_UNROLL) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int nthread = blockDim.x * blockDim.y;
  if (bid < nbatch) {
    // Shared memory with size blockDim.y * blockDim.x * sizeof(float)
    extern __shared__ char buf[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // 1. Calculate: mean(out_grad * gamma / std, axis=-1)
    //               mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    float sum_val0 = 0;  // Stores mean(out_grad * gamma / std, axis=-1)
    float sum_val1 = 0;  // Stores mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    float mean = mean_data[bid];
    float invstd_eps = float(1) / std_data[bid];
    int l = LOAD_UNROLL * tid;
    for (; l + LOAD_UNROLL - 1 < nchannel; l += nthread * LOAD_UNROLL) {
	  #pragma unroll
      for (int i = 0; i < LOAD_UNROLL; ++i) {
        float ele_og = out_grad[bid * nchannel + l + i];
        float ele_x = in_data[bid * nchannel + l + i];
        float ele_gamma = gamma[l + i];
        sum_val0 += ele_og * ele_gamma * invstd_eps;
        sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
      }
    }
    for (; l < nchannel; ++l) {
      float ele_og = out_grad[bid * nchannel + l];
      float ele_x = in_data[bid * nchannel + l];
      float ele_gamma = gamma[l];
      sum_val0 += ele_og * ele_gamma * invstd_eps;
      sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
    }
    // Intra-warp reduction (all-reduce)
    for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
      sum_val0 += warp_shfl_xor(sum_val0, mask);
      sum_val1 += warp_shfl_xor(sum_val1, mask);
    }
    // Inter-warp reduction (all-reduce)
    if (blockDim.y > 1) {
      float* sum_val0_buf = reinterpret_cast<float*>(buf);
      float* sum_val1_buf = reinterpret_cast<float*>(buf + blockDim.y / 2 * blockDim.x * sizeof(float));
      for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          sum_val0_buf[idx] = sum_val0;
          sum_val1_buf[idx] = sum_val1;
        }
        __syncthreads();
        if (threadIdx.y < offset) {
          const int idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_val0 += sum_val0_buf[idx];
          sum_val1 += sum_val1_buf[idx];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        sum_val0_buf[threadIdx.x] = sum_val0;
        sum_val1_buf[threadIdx.x] = sum_val1;
      }
      __syncthreads();
      sum_val0 = sum_val0_buf[threadIdx.x];
      sum_val1 = sum_val1_buf[threadIdx.x];
    }
    sum_val0 /= nchannel;
    sum_val1 /= nchannel;
    // 2. Calculate the gradient as
    //      out_grad * gamma / std - sum_val0 - (x - mean) / std * sum_val1
    for (int l = tid; l < nchannel; l += nthread) {
      float ele_out_grad = out_grad[bid * nchannel + l];
      float ele_x = in_data[bid * nchannel + l];
      float ele_gamma = gamma[l];
      data_grad[bid * nchannel + l] = ele_out_grad * ele_gamma * invstd_eps - sum_val0 - (ele_x - mean) * invstd_eps * sum_val1;
    }
  }
}