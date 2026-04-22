#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

extern "C"
__global__ void rope_3d_norm_igone(float* x, float* out, float* t_cos, float* t_sin, float* h_cos, float* h_sin, float* w_cos, float* w_sin, int N, int T, int headNum, int headSize, int igoneIdx)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int hhs = headSize / 2;
    int n = index / T / hhs;
    int once = index % (T * hhs);
    int t = once / hhs;
    int hs = once % hhs;
    int dim3_hs = headSize*3;
    int os = n * T * dim3_hs + t * dim3_hs;
    if(t >= igoneIdx){
		float cv = x[os + hs];
		float cn = x[os + hs + hhs];
		int rt = t - igoneIdx;
		// t
		out[os + hs] = t_cos[rt * headSize + hs] * cv - t_sin[rt * headSize + hs] * cn;
    	out[os + hs + hhs] = t_cos[rt * headSize + hs + hhs] * cn + t_sin[rt * headSize + hs + hhs] * cv;
    	// h
    	cv = x[os + headSize + hs];
    	cn = x[os + headSize + hs + hhs];
		out[os + headSize + hs] = h_cos[rt * headSize + hs] * cv - h_sin[rt * headSize + hs] * cn;
    	out[os + headSize + hs + hhs] = h_cos[rt * headSize + hs + hhs] * cn + h_sin[rt * headSize + hs + hhs] * cv;
    	// w
    	cv = x[os + headSize * 2 + hs];
    	cn = x[os + headSize * 2 + hs + hhs];
		out[os + headSize * 2 + hs] = w_cos[rt * headSize + hs] * cv - w_sin[rt * headSize + hs] * cn;
    	out[os + headSize * 2 + hs + hhs] = w_cos[rt * headSize + hs + hhs] * cn + w_sin[rt * headSize + hs + hhs] * cv;
	}else{
		out[os + hs] = x[os + hs];
		out[os + hs + hhs] = x[os + hs + hhs];
		out[os + headSize + hs] = x[os + headSize + hs];
		out[os + headSize + hs + hhs] = x[os + headSize + hs + hhs];
		out[os + headSize * 2 + hs] = x[os + headSize * 2 + hs];
		out[os + headSize * 2 + hs + hhs] = x[os + headSize * 2 + hs + hhs];
	}
}

extern "C"
__global__ void rope_3d_back_igone(float* delta, float* diff, float* t_cos, float* t_sin, float* h_cos, float* h_sin, float* w_cos, float* w_sin, int N, int T, int headNum,int headSize, int igoneIdx)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int hhs = headSize / 2;
    int n = index / T / hhs;
    int once = index % (T * hhs);
    int t = once / hhs;
    int hs = once % hhs;
    int dim3_hs = headSize*3;
    int os = n * T * dim3_hs + t * dim3_hs;
    if(t >= igoneIdx){
		float d0 = delta[os + hs];
		float d1 = delta[os + hs + hhs];
		int rt = t - igoneIdx;
		// t
		diff[os + hs] = t_cos[rt * headSize + hs] * d0 + t_sin[rt * headSize + hs + hhs] * d1;
	    diff[os + hs + hhs] = t_cos[rt * headSize + hs + hhs] * d1 - t_sin[rt * headSize + hs] * d0;
		// h
    	d0 = delta[os + headSize + hs];
    	d1 = delta[os + headSize + hs + hhs];
	    diff[os + headSize + hs] = h_cos[rt * headSize + hs] * d0 + h_sin[rt * headSize + hs + hhs] * d1;
	    diff[os + headSize + hs + hhs] = h_cos[rt * headSize + hs + hhs] * d1 - h_sin[rt * headSize + hs] * d0;
	    // w
    	d0 = delta[os + headSize * 2 + hs];
    	d1 = delta[os + headSize * 2 + hs + hhs];
    	diff[os + headSize * 2 + hs] = w_cos[rt * headSize + hs] * d0 + w_sin[rt * headSize + hs + hhs] * d1;
	    diff[os + headSize * 2 + hs + hhs] = w_cos[rt * headSize + hs + hhs] * d1 - w_sin[rt * headSize + hs] * d0;
    }else{
		diff[os + hs] = delta[os + hs];
		diff[os + hs + hhs] = delta[os + hs + hhs];
		diff[os + headSize + hs] = delta[os + headSize + hs];
		diff[os + headSize + hs + hhs] = delta[os + headSize + hs + hhs];
		diff[os + headSize * 2 + hs] = delta[os + headSize * 2 + hs];
		diff[os + headSize * 2 + hs + hhs] = delta[os + headSize * 2 + hs + hhs];
	}
}
