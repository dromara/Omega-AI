
__device__ __forceinline__ float WARP_SHFL_DOWN(float value, unsigned int delta, int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#if !defined(USE_ROCM)
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

extern "C"
__global__ void PDist_Other(const float *x, float *y, const float p, const int64_t n, const int64_t m, const float n1,
                            const float n2) {
  const int64_t pos = blockIdx.x;
  const int s = blockDim.x;

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * pos)));
  int64_t j = pos - n * i + i * (i + 1) / 2 + i + 1;

  const float *const begin = x + i * m;
  const float *const end = begin + m;
  const float *x_i = begin + threadIdx.x;
  const float *x_j = x + j * m + threadIdx.x;
  float res = 0.0;
  for (; x_i < end; x_i += s, x_j += s) {
    res += pow(abs(*x_i - *x_j), static_cast<float>(p));
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ float shared[256];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = res;
  }

  __syncthreads();
  res = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      res += WARP_SHFL_DOWN(res, offset);
    }
  }

  if (threadIdx.x == 0) {
    y[pos] = pow(res, static_cast<float>(1.0 / p));
  }
}

extern "C"
__global__ void PDist_Grad_Two(const size_t y_size, const float *y_grad, const float *x, const float *y, float *buffer, const int64_t n,
                               const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const float grad_k = y_grad[k];
  const float dist_k = y[k];

  if (dist_k != 0.0) {
    const float *const begin = x + i * m;
    const float *const end = begin + m;
    const float *x_i = begin + init;
    const float *x_j = x + j * m + init;
    float *buff1 = buffer + (ib * n + i) * m + init;
    float *buff2 = buffer + (jb * n + j) * m + init;
    for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
      float res = grad_k * (*x_i - *x_j) / dist_k;
      *buff1 = res;
      *buff2 = -res;
    }
  }
}

extern "C"
__global__ void PDist_Grad_P(const size_t y_size, const float *y_grad, const float *x, const float *y, float *buffer, const int64_t n,
                             const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const float grad_k = y_grad[k];
  const float dist_k = y[k];

  if (dist_k != 0.0) {
    const float *const begin = x + i * m;
    const float *const end = begin + m;
    const float *x_i = begin + init;
    const float *x_j = x + j * m + init;
    float *buff1 = buffer + (ib * n + i) * m + init;
    float *buff2 = buffer + (jb * n + j) * m + init;
    for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
      const float diff = (*x_i - *x_j);
      float res = diff * pow(abs(diff), static_cast<float>(p - 2)) * grad_k / pow(dist_k, static_cast<float>(p - 1));
      *buff1 = res;
      *buff2 = -res;
    }
  }
}

extern "C"
__global__ void pdist_backward_kernel_cuda_impl(float * buffer, const float * grad, const float * self, const float * dist, const int64_t n, const int64_t m, const int64_t combs, const float p,
                                                       const double n2, const double n2_squared_minus_1) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride = blockDim.y * gridDim.y;

  if (k >= combs) {
    return;
  }

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - sqrt(n2_squared_minus_1 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  printf("j:%d,", j);
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;
  printf("jb:%d,", jb);
  
  const float grad_k = grad[k];
  const float dist_k = dist[k];

  const float * const start = self + i * m;
  const float * const end = start + m;
  const float * self_i = start + init;
  const float * self_j = self + j * m + init;
  float * buff_i = buffer + (ib * n + i) * m + init;
  float * buff_j = buffer + (jb * n + j) * m + init;
  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride, buff_j += stride) {
    //const float res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    const float diff = (*self_i - *self_j);
    float res = diff * pow(abs(diff), static_cast<float>(p - 2)) * grad_k / pow(dist_k, static_cast<float>(p - 1));
    *buff_i = res;
    *buff_j = -res;
  }
}

extern "C"
__global__ void AddBuffer(float *x_grad, float *buffer, const int64_t n, const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float res = 0.0;
    float *buff = buffer + pos;
    for (int64_t i = 0; i < n - 1; ++i, buff += size) {
      res += *(buff);
    }
    x_grad[pos] = res;
  }
  return;
}

extern "C"
__global__ void log_exp_kernel(
    float* x,
    float* output,
    float* tmp,
    int N,
    int x_dim,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
	   int n = W * 2 + N;
	   float loss = 0.0f;
	   for(int i = 0;i<W;i++){
		  float d = x[i] * x[i] / x_dim;
	      loss += 2 * exp(-d) / n;
	   }
	   for(int i = 0;i<N;i++){
	      loss += expf(0) / n;
	   }
	   tmp[0] = loss;
       output[0] = log(loss);
    }
}

extern "C"
__global__ void log_exp_back_kernel(
    float* x,
    float* dx,
    float* tmp,
    int N,
    int x_dim,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(idx < W){
		int n = W * 2 + N;
		float delta = 1.0f / tmp[0] / n;
		float d = x[idx] * x[idx] / x_dim;
		float d_delta = -delta * exp(-d);
		dx[idx] = 2 * 2 * x[idx] / x_dim * d_delta;
	}
}