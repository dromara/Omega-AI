#define BLOCK 1024 


extern "C"
__global__ void add(float* a, float* b, float* output, int size)
{
    
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
	
	output[id] = a[id] + b[id];
	
}
