#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
__global__ void compute_xt(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float alpha_t = tf;
       float sigma_t = 1 - tf;
       
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_ut(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float d_alpha_t = 1;
       float d_sigma_t = -1;
       
	   output[idx] = d_alpha_t * latend[idx] + d_sigma_t * noise[idx];
    }
}

extern "C"
__global__ void cosine_similarity_loss(
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* out,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / C / W;
	   int c = idx % C;
	   int w = idx % W;
	   int x_idx = n * C * W + c * W + w;
	   int n_idx = n * W + w;
       out[idx] = 1 - (x1[x_idx] / norm1[n_idx]) * (x2[x_idx] / norm2[n_idx]);
    }
}

extern "C"
__global__ void cosine_similarity_loss_back1(
	float delta,
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* dx1,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / W;
	   int w = idx % W;
	   for(int c = 0;c<C;c++){
	      int x_idx = n * C * W + c * W + w;
	       dx1[x_idx] = -delta * (x2[x_idx] / norm2[idx]) / norm1[idx];
	   }
    }
}

extern "C"
__global__ void cosine_similarity_loss_back2(
	float delta,
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* dnorm1,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / W;
	   int w = idx % W;
	   
	   float out = 0.0f;
	   
	   for(int c = 0;c<C;c++){
	      int x_idx = n * C * W + c * W + w;
	      out += -delta * (-x1[x_idx] / norm1[idx] / norm1[idx]) * (x2[x_idx] / norm2[idx]);
	   }
	   
	   dnorm1[idx] = out;
    }
}

extern "C"
__global__ void latend_norm(
    float* x1,
    float* mean,
    float* std,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int c = idx / W % C;
	   x1[idx] = (x1[idx] - mean[c]) / std[c];
    }
}

extern "C"
__global__ void latend_un_norm(
    float* x1,
    float* mean,
    float* std,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int c = idx / W % C;
	   x1[idx] = x1[idx] * std[c] + mean[c];
    }
}