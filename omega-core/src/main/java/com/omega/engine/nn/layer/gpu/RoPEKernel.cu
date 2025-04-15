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