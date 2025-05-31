#define BLOCK 1024

extern "C"
__global__ void loss(int N,float *input, float *label, float *output, float beta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= N) return;
	float x = input[id] - label[id];
	float abs_x = abs(x);
	if(abs_x < beta){
		output[id] = 0.5f * x * x / beta;
	}else{
		output[id] = abs_x - 0.5f * beta;
	}
}

extern "C"
__global__ void loss_back(int N,float *input, float *label, float *diff, float beta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  	if (id >= N) return;
  	float x = input[id] - label[id];
	float abs_x = abs(x);
	float delta = 1.0f / N;
	if(abs_x < beta){
		diff[id] = x * delta / beta;
	}else{
		if(x > 0){
			diff[id] = 1 * delta;
		}else if(x == 0){
			diff[id] = 0;
		}else{
			diff[id] = -1 * delta;
		}
	}
}
