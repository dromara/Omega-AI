#define BLOCK 1024 


extern "C"
__global__ void add_output_bias(float* o1, float* o2, float* biases, int batch, int n, int size)
{
    
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    o1[(k*n+j)*size + i] += o2[(k*n+j)*size + i] + biases[j];
	
}

extern "C"
__global__ void add_output(float* o1, float* o2, int batch, int n, int size)
{
    
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    o1[(k*n+j)*size + i] += o2[(k*n+j)*size + i];
	
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
