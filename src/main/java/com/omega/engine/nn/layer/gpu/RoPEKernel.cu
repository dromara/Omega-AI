#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

extern "C"
__global__ void rope_norm(const float* x, float* dst,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (headSize / 2) + (col / 2) % (headSize / 2);
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

extern "C"
__global__ void rope_backward(float* delta, float* diff,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (headSize / 2) + (col / 2) % (headSize / 2);
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float d0 = delta[i + 0];
    const float d1 = delta[i + 1];

    diff[i + 0] = d0*cos_theta + d1*sin_theta;
    diff[i + 1] = d1*cos_theta - d0*sin_theta;
}

extern "C"
__global__ void rope_2d_norm(float* x, float* out,float* cos, float* sin, int N, int T, int headNum,int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index - (n * T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    float cv = x[index];
	float cn = x[index+1];
    out[index] = cos[t * headSize + hs] * cv - sin[t * headSize + hs] * cn;
    out[index+1] = cos[t * headSize + hs + 1] * cn + sin[t * headSize + hs + 1] * cv;
}

extern "C"
__global__ void rope_2d_back(float* delta, float* diff,float* cos, float* sin, int N, int T, int headNum,int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index - (n * T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    
    diff[index] = cos[t * headSize + hs] * d0 + sin[t * headSize + hs + 1] * d1;
    diff[index+1] = cos[t * headSize + hs + 1] * d1 - sin[t * headSize + hs] * d0;
}

extern "C"
__global__ void rope_2d_norm_idskeep(float* x, float* out, float *idskeep, float* cos, float* sin, int N, int T, int headNum,int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index - (n * T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    float cv = x[index];
	float cn = x[index+1];
	int t_idx = idskeep[n * T + t];
    out[index] = cos[t_idx * headSize + hs] * cv - sin[t_idx * headSize + hs] * cn;
    out[index+1] = cos[t_idx * headSize + hs + 1] * cn + sin[t_idx * headSize + hs + 1] * cv;
}

extern "C"
__global__ void rope_2d_back_idskeep(float* delta, float* diff, float *idskeep,float* cos, float* sin, int N, int T, int headNum,int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index - (n * T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    int t_idx = idskeep[n * T + t];
    diff[index] = cos[t_idx * headSize + hs] * d0 + sin[t_idx * headSize + hs + 1] * d1;
    diff[index+1] = cos[t_idx * headSize + hs + 1] * d1 - sin[t_idx * headSize + hs] * d0;
}

extern "C"
__global__ void rope_all_norm(const float* q,const float* k, float* qo, float* ko,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (headSize / 2) + (col / 2) % (headSize / 2);
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float q0 = q[i + 0];
    const float q1 = q[i + 1];
    const float k0 = k[i + 0];
    const float k1 = k[i + 1];

    qo[i + 0] = q0*cos_theta - q1*sin_theta;
    qo[i + 1] = q0*sin_theta + q1*cos_theta;
    ko[i + 0] = k0*cos_theta - k1*sin_theta;
    ko[i + 1] = k0*sin_theta + k1*cos_theta;
}

extern "C"
__global__ void rope_all_backward(float* deltaQ,float* deltaK, float* diffQ, float* diffK,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (headSize / 2) + (col / 2) % (headSize / 2);
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float dq0 = deltaQ[i + 0];
    const float dq1 = deltaQ[i + 1];
    const float dk0 = deltaK[i + 0];
    const float dk1 = deltaK[i + 1];

    diffQ[i + 0] = dq0*cos_theta + dq1*sin_theta;
    diffQ[i + 1] = dq1*cos_theta - dq0*sin_theta;
    diffK[i + 0] = dk0*cos_theta + dk1*sin_theta;
    diffK[i + 1] = dk1*cos_theta - dk0*sin_theta;
}

extern "C"
__global__ void rope_f32(const float * x, float * dst, const int ncols, const int T, const float theta_scale) {
     const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
	const int t = row % T;
    const float theta = t*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

extern "C"
__global__ void rope_backward_f32(float* delta, float* diff, const int ncols, const int T, const float theta_scale) {
     const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
	const int t = row % T;
    const float theta = t*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float d0 = delta[i + 0];
    const float d1 = delta[i + 1];
	
	diff[i + 0] = d0*cos_theta + d1*sin_theta;
    diff[i + 1] = d1*cos_theta - d0*sin_theta;

}

extern "C"
__global__ void rope_all_f32(const float * q,const float * k, float * rq, float * rk, const int ncols, const int T, const float theta_scale) {
     const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
	const int t = row % T;
    const float theta = t*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float q0 = q[i + 0];
    const float q1 = q[i + 1];
    const float k0 = k[i + 0];
    const float k1 = k[i + 1];

    rq[i + 0] = q0*cos_theta - q1*sin_theta;
    rq[i + 1] = q0*sin_theta + q1*cos_theta;
    rk[i + 0] = k0*cos_theta - k1*sin_theta;
    rk[i + 1] = k0*sin_theta + k1*cos_theta;
}

extern "C"
__global__ void rope_all_backward_f32(float* deltaQ,float* deltaK, float* diffQ, float* diffK, const int ncols, const int T, const float theta_scale) {
     const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
	const int t = row % T;
    const float theta = t*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float q0 = deltaQ[i + 0];
    const float q1 = deltaQ[i + 1];
    const float k0 = deltaK[i + 0];
    const float k1 = deltaK[i + 1];
	
	diffQ[i + 0] = q0*cos_theta + q1*sin_theta;
    diffQ[i + 1] = q1*cos_theta - q0*sin_theta;
    diffK[i + 0] = k0*cos_theta + k1*sin_theta;
    diffK[i + 1] = k1*cos_theta - k0*sin_theta;
}

extern "C"
__global__ void rope_2d_norm_igone(float* x, float* out,float* cos, float* sin, int N, int T, int headNum, int headSize, int igoneIdx)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index % (T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    float cv = x[index];
	float cn = x[index+1];
    if(t >= igoneIdx){
		int rt = t - igoneIdx;
		out[index] = cos[rt * headSize + hs] * cv - sin[rt * headSize + hs] * cn;
    	out[index+1] = cos[rt * headSize + hs + 1] * cn + sin[rt * headSize + hs + 1] * cv;
	}else{
		out[index] = cv;
    	out[index+1] = cn;
	}
}

extern "C"
__global__ void rope_2d_back_igone(float* delta, float* diff,float* cos, float* sin, int N, int T, int headNum,int headSize, int igoneIdx)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headSize;
    int once = index % (T * headSize);
    int t = once / headSize;
    int hs = once % headSize;
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    if(t >= igoneIdx){
		int rt = t - igoneIdx;
	    diff[index] = cos[rt * headSize + hs] * d0 + sin[rt * headSize + hs + 1] * d1;
	    diff[index+1] = cos[rt * headSize + hs + 1] * d1 - sin[rt * headSize + hs] * d0;
    }else{
		diff[index] = d0;
    	diff[index+1] = d1;
	}
}

extern "C"
__global__ void rope_2d_norm_t(float* x, float* out,float* cos, float* sin, int N, int T, int headNum, int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headNum / headSize;
    int once = index - (n * T * headNum * headSize);
    int t = once / headNum / headSize;
    int hs = once % (headNum * headSize) % headSize;
    float cv = x[index];
	float cn = x[index+1];
    out[index] = cos[t * headSize + hs] * cv - sin[t * headSize + hs] * cn;
    out[index+1] = cos[t * headSize + hs + 1] * cn + sin[t * headSize + hs + 1] * cv;
}

extern "C"
__global__ void rope_2d_back_t(float* delta, float* diff,float* cos, float* sin, int N, int T, int headNum, int headSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
    int index = i * 2;
    int n = index / T / headNum / headSize;
    int once = index - (n * T * headNum * headSize);
    int t = once / headNum / headSize;
    int hs = once % (headNum * headSize) % headSize;
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    diff[index] = cos[t * headSize + hs] * d0 + sin[t * headSize + hs + 1] * d1;
	diff[index+1] = cos[t * headSize + hs + 1] * d1 - sin[t * headSize + hs] * d0;
}

extern "C"
__global__ void apply_rotary_emb(const float *x, float *out, float *pos, int N, int headNum, int T, int headSize) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
	int hSize = headSize / 2;
    int n = i / T / hSize;
    int once = i - (n * T * hSize);
    int t = once / hSize;
    int hs = once % hSize * 2;
    int index = i * 2;
    float cv = x[index];
	float cn = x[index+1];
    out[index] = pos[t * headSize + hs] * cv - pos[t * headSize + hs + 1] * cn;
    out[index+1] = pos[t * headSize + hs + 1] * cv + pos[t * headSize + hs] * cn;
}

extern "C"
__global__ void apply_rotary_emb_back(const float *delta, float *dx, float *pos, int N, int headNum, int T, int headSize) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
	int hSize = headSize / 2;
    int n = i / T / hSize;
    int once = i - (n * T * hSize);
    int t = once / hSize;
    int hs = once % hSize * 2;
    int index = i * 2;
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    dx[index] = pos[t * headSize + hs] * d0 + pos[t * headSize + hs + 1] * d1;
    dx[index+1] = pos[t * headSize + hs] * d1 - pos[t * headSize + hs + 1] * d0;
}

extern "C"
__global__ void apply_rotary_emb_idskeep(const float *x, float *out, float *pos, float *idskeep, int N, int headNum, int T, int headSize) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
	int hSize = headSize / 2;
    int n = i / T / hSize;
    int once = i - (n * T * hSize);
    int t = once / hSize;
    int hs = once % hSize * 2;
    int index = i * 2;
    float cv = x[index];
	float cn = x[index+1];
	int t_idx = idskeep[t];
    out[index] = pos[t_idx * headSize + hs] * cv - pos[t_idx * headSize + hs + 1] * cn;
    out[index+1] = pos[t_idx * headSize + hs + 1] * cv + pos[t_idx * headSize + hs] * cn;
}

extern "C"
__global__ void apply_rotary_emb_back_idskeep(const float *delta, float *dx, float *pos, float *idskeep, int N, int headNum, int T, int headSize) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N/2) return;
	int hSize = headSize / 2;
    int n = i / T / hSize;
    int once = i - (n * T * hSize);
    int t = once / hSize;
    int hs = once % hSize * 2;
    int index = i * 2;
    const float d0 = delta[index + 0];
    const float d1 = delta[index + 1];
    int t_idx = idskeep[t];
    dx[index] = pos[t_idx * headSize + hs] * d0 + pos[t_idx * headSize + hs + 1] * d1;
    dx[index+1] = pos[t_idx * headSize + hs] * d1 - pos[t_idx * headSize + hs + 1] * d0;
}