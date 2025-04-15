#define BLOCK 1024 

extern "C"
__global__ void lpip_l2_kernel(float *x1,float *x2, float *out, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N) out[id] = powf(x1[id] - x2[id], 2);
}

extern "C"
__global__ void lpip_l2_backward_kernel(float *delta,float *x1,float *x2, float *diff, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N) diff[id] = delta[id] * 2 * (x1[id] - x2[id]);
}

extern "C"
__global__ void scaling_kernel(int N,float *x,float *shift, float *scale, float *out, int C,int HW)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){ 
    	int c = id / HW % C;
    	out[id] = (x[id] - shift[c]) / scale[c];
    }
}

extern "C"
__global__ void scaling_back_kernel(int N,float *dy, float *scale, float *dx, int C,int HW)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){ 
    	int c = id / HW % C;
    	dx[id] = dy[id] / scale[c];
    }
}