
extern "C"
__global__ void vectorL2NormKernel(const float* a, float* result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float cache[512]; // Assuming a block size of 512, adjust as needed

    float temp_sum = 0.0;
    for (int i = index; i < n; i += stride) {
        temp_sum += a[i] * a[i];
    }

    cache[threadIdx.x] = temp_sum;
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
        atomicAdd(result, cache[0]);
}


extern "C"
__global__ void l2NormKernel(const float* d_vector, float* d_partialSum, int size) {
    extern __shared__ float sharedData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (i < size) ? d_vector[i] * d_vector[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
}