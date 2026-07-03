#define BLOCK 1024

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n,float eta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
		int index = id * n + i;
        sum += (label[index] * logf(input[index] + eta) + (1 - label[index]) * logf((1 - (1 / (1 + expf(-input[index]))))));
    }
    
	output[id] = sum / n;
}

extern "C"
__global__ void loss_back(float *x, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = (1 / (1 + expf(-x[id])) - currentLabel[id]) / batch / n;    

}
