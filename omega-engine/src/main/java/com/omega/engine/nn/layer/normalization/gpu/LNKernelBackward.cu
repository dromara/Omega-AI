
extern "C"
__device__ __inline__ float get_value(const float* index, const int bound_check,
                                  const int up_bound) {
  if (bound_check < up_bound)
    return __ldg(index);
  else
    return 0.0f;
}

extern "C"
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

extern "C"
__device__ __inline__ float warpSum2(float val1) {
  	val1 += __shfl_xor(val1, 16);
  	val1 += __shfl_xor(val1, 8);
  	val1 += __shfl_xor(val1, 4);
  	val1 += __shfl_xor(val1, 2);
  	val1 += __shfl_xor(val1, 1);
  	return val1;
}


extern "C"
__global__ void LayerNorm1FusedBackpropGPUKernel(const int in_depth,const int n_inputs,const float epsilon,
    const float* __restrict__ input,
    const float* __restrict__ out_back,
    const float* __restrict__ gamma,
    float* __restrict__ in_back,
    float* __restrict__ gamma_back,
    float* __restrict__ beta_back, 
    const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  float* dmu_cache = (float*)&my_smem[2 * sizeof(float)];
  float* dstd_cache = (float*)&my_smem[3 * sizeof(float)];

  const int mult = 1;

  float inp[mult];
  float dout[mult];

  float _gamma[mult];
  float _gamma_bp[mult];
  float _beta_bp[mult];
  int thread_id[mult];

  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _beta_bp[m] = 0.0f;
    _gamma_bp[m] = 0.0f;
    _gamma[m] = get_value(gamma + threadIdx.x + m * blockDim.x,
                             threadIdx.x + m * blockDim.x, in_depth);
  }

  float mu;
  float rstd;
  float dstd;
  float dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    if (threadIdx.x == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
      dmu_cache[0] = 0.0f;
      dstd_cache[0] = 0.0f;
    }
    __syncthreads();
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], threadIdx.x + m * blockDim.x,
                            in_depth);
      dout[m] = get_value(out_back + thread_id[m],
                             threadIdx.x + m * blockDim.x, in_depth);
    }

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      _beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * _gamma[m] * i_n;
    }
	
	mu = warpSum2(mu);
	dmu = warpSum2(dmu);
	
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], mu);
      atomicAdd(&dmu_cache[0], dmu);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * _gamma[m];
      }
    }

	rstd = warpSum2(rstd);
	dstd = warpSum2(dstd);

    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], rstd);
      atomicAdd(&dstd_cache[0], dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(std_cache[0] * i_n + epsilon);
      std_cache[0] = rstd;
      dmu_cache[0] = dmu_cache[0] * rstd;
      dstd_cache[0] = dstd_cache[0] * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = std_cache[0];
    dstd = dstd_cache[0];
    dmu = dmu_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        _gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] =
            dout[m] * _gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, _gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, _beta_bp[m]);
    }
  }
}

extern "C"
__global__ void LayerNorm2FusedBackpropGPUKernel(const int in_depth,const int n_inputs,const float epsilon,
    const float* __restrict__ input,
    const float* __restrict__ out_back,
    const float* __restrict__ gamma,
    float* __restrict__ in_back,
    float* __restrict__ gamma_back,
    float* __restrict__ beta_back, 
    const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  float* dmu_cache = (float*)&my_smem[2 * sizeof(float)];
  float* dstd_cache = (float*)&my_smem[3 * sizeof(float)];

  const int mult = 2;

  float inp[mult];
  float dout[mult];

  float _gamma[mult];
  float _gamma_bp[mult];
  float _beta_bp[mult];
  int thread_id[mult];

  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _beta_bp[m] = 0.0f;
    _gamma_bp[m] = 0.0f;
    _gamma[m] = get_value(gamma + threadIdx.x + m * blockDim.x,
                             threadIdx.x + m * blockDim.x, in_depth);
  }

  float mu;
  float rstd;
  float dstd;
  float dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    if (threadIdx.x == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
      dmu_cache[0] = 0.0f;
      dstd_cache[0] = 0.0f;
    }
    __syncthreads();
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], threadIdx.x + m * blockDim.x,
                            in_depth);
      dout[m] = get_value(out_back + thread_id[m],
                             threadIdx.x + m * blockDim.x, in_depth);
    }

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      _beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * _gamma[m] * i_n;
    }

    mu = warpSum2(mu);
	dmu = warpSum2(dmu);
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], mu);
      atomicAdd(&dmu_cache[0], dmu);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * _gamma[m];
      }
    }

    rstd = warpSum2(rstd);
	dstd = warpSum2(dstd);

    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], rstd);
      atomicAdd(&dstd_cache[0], dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(std_cache[0] * i_n + epsilon);
      std_cache[0] = rstd;
      dmu_cache[0] = dmu_cache[0] * rstd;
      dstd_cache[0] = dstd_cache[0] * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = std_cache[0];
    dstd = dstd_cache[0];
    dmu = dmu_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        _gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] =
            dout[m] * _gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, _gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, _beta_bp[m]);
    }
  }
}

extern "C"
__global__ void LayerNorm3FusedBackpropGPUKernel(const int in_depth,const int n_inputs,const float epsilon,
    const float* __restrict__ input,
    const float* __restrict__ out_back,
    const float* __restrict__ gamma,
    float* __restrict__ in_back,
    float* __restrict__ gamma_back,
    float* __restrict__ beta_back, 
    const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  float* dmu_cache = (float*)&my_smem[2 * sizeof(float)];
  float* dstd_cache = (float*)&my_smem[3 * sizeof(float)];

  const int mult = 3;

  float inp[mult];
  float dout[mult];

  float _gamma[mult];
  float _gamma_bp[mult];
  float _beta_bp[mult];
  int thread_id[mult];

  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _beta_bp[m] = 0.0f;
    _gamma_bp[m] = 0.0f;
    _gamma[m] = get_value(gamma + threadIdx.x + m * blockDim.x,
                             threadIdx.x + m * blockDim.x, in_depth);
  }

  float mu;
  float rstd;
  float dstd;
  float dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    if (threadIdx.x == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
      dmu_cache[0] = 0.0f;
      dstd_cache[0] = 0.0f;
    }
    __syncthreads();
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], threadIdx.x + m * blockDim.x,
                            in_depth);
      dout[m] = get_value(out_back + thread_id[m],
                             threadIdx.x + m * blockDim.x, in_depth);
    }

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      _beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * _gamma[m] * i_n;
    }

    mu = warpSum2(mu);
	dmu = warpSum2(dmu);
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], mu);
      atomicAdd(&dmu_cache[0], dmu);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * _gamma[m];
      }
    }

    rstd = warpSum2(rstd);
	dstd = warpSum2(dstd);

    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], rstd);
      atomicAdd(&dstd_cache[0], dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(std_cache[0] * i_n + epsilon);
      std_cache[0] = rstd;
      dmu_cache[0] = dmu_cache[0] * rstd;
      dstd_cache[0] = dstd_cache[0] * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = std_cache[0];
    dstd = dstd_cache[0];
    dmu = dmu_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        _gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] =
            dout[m] * _gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, _gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, _beta_bp[m]);
    }
  }
}

extern "C"
__global__ void LayerNorm4FusedBackpropGPUKernel(const int in_depth,const int n_inputs,const float epsilon,
    const float* __restrict__ input,
    const float* __restrict__ out_back,
    const float* __restrict__ gamma,
    float* __restrict__ in_back,
    float* __restrict__ gamma_back,
    float* __restrict__ beta_back, 
    const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  float* dmu_cache = (float*)&my_smem[2 * sizeof(float)];
  float* dstd_cache = (float*)&my_smem[3 * sizeof(float)];

  const int mult = 4;

  float inp[mult];
  float dout[mult];

  float _gamma[mult];
  float _gamma_bp[mult];
  float _beta_bp[mult];
  int thread_id[mult];

  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _beta_bp[m] = 0.0f;
    _gamma_bp[m] = 0.0f;
    _gamma[m] = get_value(gamma + threadIdx.x + m * blockDim.x,
                             threadIdx.x + m * blockDim.x, in_depth);
  }

  float mu;
  float rstd;
  float dstd;
  float dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    if (threadIdx.x == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
      dmu_cache[0] = 0.0f;
      dstd_cache[0] = 0.0f;
    }
    __syncthreads();
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], threadIdx.x + m * blockDim.x,
                            in_depth);
      dout[m] = get_value(out_back + thread_id[m],
                             threadIdx.x + m * blockDim.x, in_depth);
    }

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      _beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * _gamma[m] * i_n;
    }

    mu = warpSum2(mu);
	dmu = warpSum2(dmu);
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], mu);
      atomicAdd(&dmu_cache[0], dmu);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * _gamma[m];
      }
    }

    rstd = warpSum2(rstd);
	dstd = warpSum2(dstd);

    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], rstd);
      atomicAdd(&dstd_cache[0], dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(std_cache[0] * i_n + epsilon);
      std_cache[0] = rstd;
      dmu_cache[0] = dmu_cache[0] * rstd;
      dstd_cache[0] = dstd_cache[0] * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = std_cache[0];
    dstd = dstd_cache[0];
    dmu = dmu_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        _gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] =
            dout[m] * _gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, _gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, _beta_bp[m]);
    }
  }
}

extern "C"
__global__ void LayerNorm5FusedBackpropGPUKernel(const int in_depth,const int n_inputs,const float epsilon,
    const float* __restrict__ input,
    const float* __restrict__ out_back,
    const float* __restrict__ gamma,
    float* __restrict__ in_back,
    float* __restrict__ gamma_back,
    float* __restrict__ beta_back, 
    const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  float* dmu_cache = (float*)&my_smem[2 * sizeof(float)];
  float* dstd_cache = (float*)&my_smem[3 * sizeof(float)];

  const int mult = 5;

  float inp[mult];
  float dout[mult];

  float _gamma[mult];
  float _gamma_bp[mult];
  float _beta_bp[mult];
  int thread_id[mult];

  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _beta_bp[m] = 0.0f;
    _gamma_bp[m] = 0.0f;
    _gamma[m] = get_value(gamma + threadIdx.x + m * blockDim.x, threadIdx.x + m * blockDim.x, in_depth);
  }

  float mu;
  float rstd;
  float dstd;
  float dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    if (threadIdx.x == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
      dmu_cache[0] = 0.0f;
      dstd_cache[0] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], threadIdx.x + m * blockDim.x, in_depth);
      dout[m] = get_value(out_back + thread_id[m], threadIdx.x + m * blockDim.x, in_depth);
    }

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      _beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * _gamma[m] * i_n;
    }

    mu = warpSum2(mu);
	dmu = warpSum2(dmu);
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], mu);
      atomicAdd(&dmu_cache[0], dmu);
    }
    __syncthreads();

    mu = mean_cache[0];
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * _gamma[m];
      }
    }

    rstd = warpSum2(rstd);
	dstd = warpSum2(dstd);
    
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], rstd);
      atomicAdd(&dstd_cache[0], dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(std_cache[0] * i_n + epsilon);
      std_cache[0] = rstd;
      dmu_cache[0] = dmu_cache[0] * rstd;
      dstd_cache[0] = dstd_cache[0] * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = std_cache[0];
    dstd = dstd_cache[0];
    dmu = dmu_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        _gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] = dout[m] * _gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    
    __syncthreads();
  }
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, _gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, _beta_bp[m]);
    }
  }
  
}


extern "C"
__global__ void LayerNormFusedSmallBackpropGPUKernel(
    const int slice_size,const int in_depth,const int n_inputs,const float epsilon, const float* __restrict__ input,
    const float* __restrict__ out_back, const float* __restrict__ gamma,
    float* __restrict__ in_back, float* __restrict__ gamma_back,
    float* __restrict__ beta_back, const int num_blocks,
    const int slice_per_block) {

  const int slice_id = threadIdx.x / slice_size;
  const int tSliceIdx = threadIdx.x % slice_size;
  const int tWarpIdx = threadIdx.x % warpSize;

  const float i_n = 1.0f / in_depth;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* gamma_cache = (float*)my_smem;
  float* beta_cache = (float*)&my_smem[in_depth * sizeof(float)];
  // initialize shared memory cache to 0.0
  if (threadIdx.x < in_depth) {
    gamma_cache[threadIdx.x] = 0.0f;
    beta_cache[threadIdx.x] = 0.0f;
  }

  const float _gamma = get_value(gamma + tSliceIdx, tSliceIdx, in_depth);
  float mu;
  float rstd;
  float dstd;
  float dmu;

  float _gamma_bp = 0.0f;
  float _beta_bp = 0.0f;
  // we need a thread block here to ensure initialization is complete
  __syncthreads();
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = 0.0f;
    rstd = 0.0f;
    dmu = 0.0f;
    dstd = 0.0f;

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + tSliceIdx;
    const float inp = get_value(input + thread_id, tSliceIdx, in_depth);
    const float dout = get_value(out_back + thread_id, tSliceIdx, in_depth);

    const float dout_g = dout * _gamma;
    _beta_bp += dout;
    mu += inp * i_n;
    dmu += dout * _gamma * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
      dmu += __shfl_xor(dmu, mask);
    }

    if (tSliceIdx < in_depth) {
      rstd += (inp - mu) * (inp - mu);
      dstd += (inp - mu) * dout_g;
    }

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
      dstd += __shfl_xor(dstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);
    dmu = dmu * rstd;
    dstd = dstd * rstd * rstd * rstd * i_n;

    if (tSliceIdx < in_depth && thread_id < n_inputs) {
      _gamma_bp += dout * (inp - mu) * rstd;
      in_back[thread_id] = dout_g * rstd - (inp - mu) * dstd - dmu;
    }
  }
  for (int mask = slice_size; mask < warpSize; mask *= 2) {
    _gamma_bp += __shfl_xor(_gamma_bp, mask);
    _beta_bp += __shfl_xor(_beta_bp, mask);
  }
  // accumulate *_bp into shared memory.
  if (tWarpIdx < in_depth) {
    atomicAdd(gamma_cache + tSliceIdx, _gamma_bp);
    atomicAdd(beta_cache + tSliceIdx, _beta_bp);
  }
  
  // add *_bp into global memory.
  __syncthreads();
  if (slice_id == 0 && tSliceIdx < in_depth) {
    atomicAdd(gamma_back + tSliceIdx, gamma_cache[tSliceIdx]);
    atomicAdd(beta_back + tSliceIdx, beta_cache[tSliceIdx]);
  }
  
}
