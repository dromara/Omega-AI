#define BLOCK 1024

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += powf(input[id * n + i] - label[id * n + i], 2);
    }
    
	output[id] = sum / n;
}

extern "C"
__global__ void loss_back(float *output, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
 
	diff[id] = 2 * (output[id] - currentLabel[id]) / batch / n;    

}
