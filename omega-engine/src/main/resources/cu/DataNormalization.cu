#define BLOCK 1024 
#define ETA 1e-8 

extern "C"
__global__ void normalization(float *input, float *mean, float *std, int N, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    input[index] = (input[index] / 255.0 - mean[f]) / std[f];
}
