#define BLOCK 1024 
#define forward_threads 256
#define FLT_MAX 3.402823466e+38F
#define _USE_MATH_DEFINES
#include <math.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__device__ __forceinline__ float WARP_SHFL_DOWN(float value, unsigned int delta, int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#if !defined(USE_ROCM)
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

extern "C"
__global__ void reparameterize_forward(float *mu,float *logvar,float *eps, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = eps[i] * expf(0.5f * logvar[i]) + mu[i];
    }
}

extern "C"
__global__ void reparameterize_backward(float *dmu,float *dlogvar,float *eps,float *logvar, float *delta, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	dlogvar[i] += delta[i] * eps[i] * expf(0.5f * logvar[i]) * 0.5f;
    	dmu[i] += delta[i];
    }
}


extern "C"
__global__ void kl_loss(float *mu,float *logvar,float kl_weight, float *klLoss, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	klLoss[i] = -0.5f * (1 + logvar[i] - powf(mu[i], 2) - expf(logvar[i])) * kl_weight;
    }
}

extern "C"
__global__ void kl_loss_back(float *mu,float *logvar,float kl_weight, float *dmu, float * dlogvar,int batch, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	dmu[i] = kl_weight * mu[i];
    	dlogvar[i] = kl_weight * -0.5f * (1.0f - expf(logvar[i]));
    }
}

extern "C"
__global__ void CdistP(float *x1, float *x2, float *result, double p, const int64_t r2, const int64_t m, const int64_t r_size,
                       const int64_t l1_size, const int64_t l2_size) {
  
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const float *const start = x1 + l * l1_size + i * m;
  const float *const end = start + m;
  const float *a = start + threadIdx.x;
  const float *b = x2 + l * l2_size + j * m + threadIdx.x;
  float res = 0.0;
  for (; a < end; a += stride, b += stride) {
    res += static_cast<float>(pow(static_cast<double>(abs(*a - *b)), p));
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ float shared[forward_threads];
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
    result[blockIdx.x] = static_cast<float>(pow(static_cast<double>(res), 1.0 / p));
  }
  return;
}

extern "C"
__global__ void calcDistKernel(float* _res, const float * _A, const float * _B,
		int _Arows, int _Brows, int _dim) {

	extern __shared__ float shm[];

	float *Ablock = shm;
	float *Bblock = shm + blockDim.x * blockDim.y;
	float *AB = Bblock + blockDim.x * blockDim.y;

	int id = threadIdx.x + threadIdx.y * blockDim.x;
	int arow = threadIdx.y + blockIdx.y * blockDim.y;
	int brow = threadIdx.y + blockIdx.x * blockDim.x;
	int ocol = threadIdx.x + blockIdx.x * blockDim.x;

	int AOffs = threadIdx.y * blockDim.x;
	int BOffs = threadIdx.x * blockDim.x;

	AB[id] = 0.;

	int j = blockIdx.z;
	{
//	for (int j = 0; j < _Acols; j += blockDim.x) {
		// load block of A and B
		int col = threadIdx.x + j * blockDim.x;

		Bblock[id] = 0.;
		Ablock[id] = 0.;
		if (col < _dim) {
			if (brow < _Brows)
				Bblock[id] = _B[brow * _dim + col];
			if (arow < _Arows)
				Ablock[id] = _A[arow * _dim + col];
		}
		__syncthreads();

//		if ((col < _Acols) && (arow < _Arows))
//			printf(" A B %i %f %f \n", id, Ablock[id], Bblock[id]);

		// compute partial differences
		for (int i = 0; i < blockDim.x; i++) {
			AB[id] += sqrtf(Ablock[AOffs + i] - Bblock[BOffs + i]);
		}
		__syncthreads();

	}

// write out the result
	if ((arow < _Arows) && (ocol < _Brows)) {
		//	_res[arow][ocol] += AB[id];
		atomicAdd(_res + (arow * _Brows + ocol), AB[id]);
//			printf(" AB %i %i %i %f \n", id, arow, ocol, AB[id]);
	}

}

extern "C"
__global__ void argmin(float *x,float *y,int batch, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < batch) {
    	float min = FLT_MAX;
    	int idx = 0;
    	for(int j = 0;j<n;j++){
    		if(min > x[i * n + j]){
    			min = x[i * n + j];
    			idx = j;
    		}
    	}
    	y[i] = idx;
    }
}

extern "C"
__global__ void mean_kernel(float *x,float *y,int batch)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < batch) {
    	int index = x[i];
    	y[index] += 1 / batch;
    }
}

extern "C"
__global__ void mse_loss_kernel(const float* output, const float* target, float* loss, float beta, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
      float err = output[idx] - target[idx];
      float err2 = err * err;
      float mean_square_error = err2/num_elem;
      mean_square_error = mean_square_error + beta * mean_square_error;
      atomicAdd(loss, mean_square_error); // poor performance
  }
}

extern "C"
__global__ void mse_loss_kernel_only_c(const float* output, const float* target, float* loss, float beta, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
      float err = output[idx] - target[idx];
      float err2 = err * err;
      float mean_square_error = err2/num_elem;
      mean_square_error = beta * mean_square_error;
      atomicAdd(loss, mean_square_error); // poor performance
  }
}

extern "C"
__global__ void mse_loss_back(float *x, float *y, float beta, float *diffX,float *diffY, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < n){
    	float tmp = 2 * (x[id] - y[id]) / n;
		diffX[id] = tmp;
		diffY[id] = - tmp * beta;
    }
 	
}

extern "C"
__global__ void mse_sum_loss_kernel(const float* output, const float* target, float* loss, float beta, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
      float err = output[idx] - target[idx];
      float err2 = err * err;
      float mean_square_error = err2;
      mean_square_error = mean_square_error + beta * mean_square_error;
      atomicAdd(loss, mean_square_error); // poor performance
  }
}

extern "C"
__global__ void mse_sum_loss_back(float *x, float *y, float beta, float *diffX,float *diffY, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < n){
    	float tmp = 2 * (x[id] - y[id]);
		diffX[id] = tmp;
		diffY[id] = - tmp * beta;
    }
 	
}

extern "C"
__global__ void mse_sum_only_c_loss_kernel(const float* output, const float* target, float* loss, float beta, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
      float err = output[idx] - target[idx];
      float err2 = err * err;
      float mean_square_error = err2;
      mean_square_error = beta * mean_square_error;
      atomicAdd(loss, mean_square_error); // poor performance
  }
}

extern "C"
__global__ void mse_sum_only_c_loss_back(float *x, float *y, float beta,float *diffY, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < n){
    	float tmp = 2 * (x[id] - y[id]);
		diffY[id] = - tmp * beta;
    }
 	
}

extern "C"
__global__ void ema_count(int n, float *x, float *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id == 0){
    	
	    for(int i=0;i<n;i++){
	    	int idx = (int)x[i];
			y[idx] += 1;
	    }
    	
    }
    
}

extern "C"
__global__ void move_ema_count(float *x,float *y, float decay, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < n){
		y[id] = decay * y[id] + (1 - decay) * x[id];
    }
 	
}

extern "C"
__global__ void move_ema_count2(float *x, float *sumec,float eps, int D, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < n){
		x[id] = (x[id]+ eps) / (sumec[0] + D * eps) * sumec[0];
    }
 	
}

extern "C"
__global__ void update_emb_weight(float *dw,float *weight, float *emb_weight,float *ema_count, float decay,int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
    if(id < batch){
		
		for(int i = 0;i<n;i++){
			emb_weight[id * n + i] = decay * emb_weight[id * n + i] + (1 - decay) * dw[id * n + i];
			weight[id * n + i] = emb_weight[id * n + i] / ema_count[id];
		}
		
    }
 	
}
