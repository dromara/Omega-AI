#define BLOCK 1024 

extern "C"
__global__ void add_full_bias(float* output, float* biases, int N, int w)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = i % w;
    output[i] += biases[j];
}

extern "C"
__global__ void add_bias(float* output, float* biases, int batch, int n, int size)
{
    
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
	
}

extern "C"
__global__ void add_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases[filter];
}

extern "C"
__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int W)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= W) return;
    float sum = 0;
    for(int b = 0; b < batch; ++b){
        int i = b*W + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

extern "C"
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}