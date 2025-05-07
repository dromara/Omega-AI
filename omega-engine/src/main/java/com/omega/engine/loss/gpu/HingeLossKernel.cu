#define BLOCK 1024 

__device__  float relu(float val){
    if(val > 1){
    	return val;
    }else{
    	return 0;
    }
}

extern "C"
__global__ void hinge_d_loss_kernel(float *real, float *fake,float *out, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	out[id] = relu(1 - real[id]) + relu(1 + fake[id]) / N * 0.5f;
    }
}

extern "C"
__global__ void hinge_d_loss_back_kernel(float *real, float *fake,float *dreal,float * dfake, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	if(1 - real[id] > 0){
    		dreal[id] = -1 / N * 0.5f;
    	}
    	
    	if(1 + fake[id] > 0){
    		dfake[id] = 1 / N * 0.5f;
    	}
    }
}

extern "C"
__global__ void hinge_d_real_loss_kernel(float *real,float *out, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	out[id] = relu(1 - real[id]) / N * 0.5f;
    }
}

extern "C"
__global__ void hinge_d_fake_loss_kernel(float *fake,float *out, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	out[id] = relu(1 + fake[id]) / N * 0.5f;
    }
}

extern "C"
__global__ void hinge_d_loss_real_back_kernel(float *real,float *dreal,float weight, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	if(1 - real[id] > 0){
    		dreal[id] = -1 / N * 0.5f * weight;
    	}
    }
}

extern "C"
__global__ void hinge_d_loss_fake_back_kernel(float *fake,float *dfake,float weight, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N){
    	if(1 + fake[id] > 0){
    		dfake[id] = 1 / N * 0.5f * weight;
    	}
    }
}