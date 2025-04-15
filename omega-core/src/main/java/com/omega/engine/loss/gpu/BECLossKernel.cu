#define BLOCK 1024

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n,float eta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;
	float neg_100 = -100;
	float one = 1;
	float EPSILON = 1e-12f;

	for(int i = 0;i<n;i++){
        float log_input_val = logf(input[id * n + i]);
        float log_1_minus_input_val = logf(1 - input[id * n + i]);
        log_input_val = fmax(log_input_val, neg_100);
        log_1_minus_input_val = fmax(log_1_minus_input_val, neg_100);
        sum += ((label[id * n + i] - one) * log_1_minus_input_val) - (label[id * n + i] * log_input_val);
    }
    
	output[id] = sum / batch / n;
}

extern "C"
__global__ void loss_back(float *input, float *currentLabel, float *diff, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float one = 1;
	float EPSILON = 1e-12f;
	
	for(int i = 0;i<n;i++){
		
		float grad_input_denominator = fmax((one - input[id * n + i]) * input[id * n + i], EPSILON);
	
		diff[id * n + i] = (input[id * n + i] - currentLabel[id * n + i]) / grad_input_denominator;
	}
	
}
