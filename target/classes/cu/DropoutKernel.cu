#define BLOCK 1024 

extern "C"
__global__ void forward_kernel(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

extern "C"
__global__ void dropout_kernel(float *input,float *output, float *mask, int size, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) output[id] = (mask[id] < prob) ? 0 : input[id]*scale;
}

extern "C"
__global__ void backward_kernel(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

extern "C"
__global__ void DropoutForward(const int n, const float* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    float* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n); i += blockDim.x * gridDim.x){
    	out[i] = in[i] * (mask[i] > threshold) * scale;
    }
}