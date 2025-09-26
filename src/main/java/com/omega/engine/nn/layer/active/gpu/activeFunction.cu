#define BLOCK 1024 

#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_fp16.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
#define SQRT_1_2  0.70710678118654757274f  // sqrt(1/2)


extern "C"
__global__ void relu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		output[i] = x[i];
    	}else{
    		output[i] = 0;
    	}
    }
}

extern "C"
__global__ void relu_backward(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] = delta[i];
    	}else{
    		diff[i] = 0;
    	}
    }
}

extern "C"
__global__ void relu_backward_temp(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] += delta[i];
    	}else{
    		diff[i] += 0;
    	}
    }
}

extern "C"
__global__ void leakyRelu_forward(float *x, float *output, int n,float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		output[i] = x[i];
    	}else{
    		output[i] = x[i] * s;
    	}
    }
}

extern "C"
__global__ void leakyRelu_backward(float *x, float *delta, float *diff, int n,float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] = delta[i];
    	}else{
    		diff[i] = delta[i] * s;
    	}
    }
}

extern "C"
__global__ void leakyRelu_backward_temp(float *x, float *delta, float *diff, int n, float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] += delta[i];
    	}else{
    		diff[i] += delta[i] * s;
    	}
    }
}

extern "C"
__global__ void sigmod_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = (float) (1.0f / (1.0f + expf(-x[i])));
    }
}

extern "C"
__global__ void sigmod_backward(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] = delta[i] * output[i] * (1.0f - output[i]);
    }
}

extern "C"
__global__ void swish_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = (float) (1.0f / (1.0f + expf(-x[i]))) * x[i];
    }
}

extern "C"
__global__ void swish_backward(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
		float so = (float) (1.0f / (1.0f + expf(-x[i])));
		float dx_dy = so * delta[i];
		float dx_ds = x[i] * delta[i] * so * (1.0f - so);
    	diff[i] = dx_dy + dx_ds;
    }
}

extern "C"
__global__ void sigmod_backward_temp(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * output[i] * (1.0f - output[i]);
    }
}

extern "C"
__global__ void tanh_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	//float e = expf(-2 * x[i]);
    	//output[i] = (1 - e) / (1 + e);
    	float x1 = expf(x[i]);
    	float x2 = expf(-x[i]);
    	output[i] = (x1 - x2) / (x1 + x2);
    }
}

extern "C"
__global__ void tanh_backward(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] = delta[i] * (1.0f - output[i]  * output[i]);
    }
}

extern "C"
__global__ void tanh_backward_temp(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * (1.0f - output[i]  * output[i]);
    }
}

extern "C"
__global__ void silu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

extern "C"
__global__ void silu_backward(float *x, float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	float x_i = x[i];
    	float x_sigmod = (float) (1.0f / (1.0f + expf(-x_i)));
    	diff[i] = delta[i] * (x_sigmod * (1 + x_i * (1 - x_sigmod)));
    }
}

extern "C"
__global__ void silu_backward_temp(float *x, float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * (output[i] + 1.0f / (1.0f + expf(-x[i])) * (1.0f - output[i]));
    }
}

extern "C"
__device__ float sech_gpu(float x) {
  return 2 / (expf(x) + expf(-x)); 
}

__device__ float sigmoid(float x) {
    return 1.0/(1+expf(-x));
}

extern "C"
__global__ void gelu_fwd_cuda(float *x, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n) {
        output[idx] = x[idx]*sigmoid(1.702*x[idx]);
    }
}

extern "C"
__global__ void gelu_bwd_cuda(float *x, float *delta, float *diff, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n) {
        float tmp = sigmoid(1.702*x[idx]);
        diff[idx] = delta[idx]*(tmp + 1.702*x[idx]*tmp*(1-tmp));
    }
}

extern "C"
__global__ void gelu_forward(float *x, float *out, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N) {
    	float xi = x[idx];
    	float cube = 0.044715f * xi * xi * xi;
        out[idx] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

extern "C"
__global__ void gelu_old_forward(float *x, float *out, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N) {
    	float xi = x[idx];
    	out[idx] = xi * 0.5f * (1.0f + erff(xi * SQRT_1_2));
    }
}

extern "C"
__global__ void gelu_backward(float* dinp, const float* inp, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

extern "C"
__global__ void gelu_old_half_forward(float *x, float *out, int N) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(idx < N) {
    	float val = x[idx];
    	float v = __float2half(erff(val * 0.707106781f));
    	out[idx] = val * 0.5 * (1 + v);
    }
}