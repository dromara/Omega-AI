#define BLOCK 1024

extern "C"
__global__ void norm(float *X, float *Y, int N)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i == 0){
    	Y[0] = 0.0f;
    }
    if(i < 1) {
	    for(int index = 0;index<N;index++){
	    	Y[0] += powf(X[index], 2);
	    }
    }
    __syncthreads();
    if(i == 0){
    	Y[0] = sqrtf(Y[0]);
    }
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

extern "C"
__global__ void l2norm_kernel(int N, float *x,float *out, int filters)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters + f;
        sum += powf(x[index], 2);
    }
    sum = sqrtf(sum);
    if(sum == 0) sum = 1;
    //printf("%f\n", sum);
    for(f = 0; f < filters; ++f){
        int index = b*filters + f;
        out[index] = x[index] / sum;
    }
}

extern "C"
__global__ void l2norm_backward_kernel(int N, float *x,float *out,float *delta, float *dx, int filters)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters + f;
        sum += powf(x[index], 2);
    }
    float s = sqrtf(sum);
    for(f = 0; f < filters; ++f){
        int index = b*filters + f;
    	dx[index] = delta[index] / s + (-powf(sum, -0.5) * x[index]);
    }
}


extern "C"
__global__ void l2norm_1dim_kernel(int N, float *x,float *out, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        float v = x[index];
        sum += v * v;
    }
    
    float norm = 1.0 / (sqrtf(sum) + eps);
    
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        out[index] = x[index] * norm;
    }
}

extern "C"
__global__ void l2norm_1dim_backward_kernel(int N, float *x,float *out,float *delta, float *dx, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    float dsum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
       	float v = x[index];
        sum += v * v;
        dsum += out[index] * delta[index];
    }
    float norm = rsqrtf(fmaxf(sum, eps));
    if(sum >= eps){
    	for(f = 0; f < filters; ++f){
	        int index = b*filters*spatial + f*spatial + i;
	        dx[index] = norm * (delta[index] - (out[index] * dsum));
    	}
    }else{
    	for(f = 0; f < filters; ++f){
	        int index = b*filters*spatial + f*spatial + i;
	        dx[index] = norm * delta[index];
    	}
    }
    
}

extern "C"
__global__ void l2norm_1dim_backward_kernel2(int N, float *x,float *delta, float *dx, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    float dsum = 0;
    int offset = b*filters*spatial + i;
    for(f = 0; f < filters; ++f){
        int index = offset + f*spatial;
        float v = x[index];
        sum += v * v;
        dsum += v * delta[index];
    }
    float norm = 1.0 / (sqrtf(sum) + eps);
    for(f = 0; f < filters; ++f){
        int index = offset + f*spatial;
        dx[index] = norm * delta[index] - norm * dsum / sum * x[index];
    }
}

extern "C"
__global__ void l2norm_1dim_backward_kernel3(int N, float *x,float *delta, float *dx, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    int offset = b*filters*spatial + i;
    for(f = 0; f < filters; ++f){
        int index = offset + f*spatial;
        float v = x[index];
        sum += powf(v, 2);
    }
    float sq = sqrtf(sum) + eps;

    float dnorm = 0.0;
    for(f = 0; f < filters; ++f){
		int index = offset + f*spatial;
		dnorm += -delta[index] * x[index] / sq / sq;
	}
	//printf("norm:%f", norm);
	float dsq = powf(sum, -0.5) * dnorm;
   // printf("dnorm:%f", dsq);
    for(f = 0; f < filters; ++f){
        int index = offset + f*spatial;
        dx[index] = delta[index] / sq + dsq * x[index];
        //printf("bx:%f", dsq * 2 * x[index]);
        //printf("dx2:%f", dx2);
    }
}


extern "C"
__global__ void l2norm_1dim_kernel2(int N, float *x,float *out, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int x_index = b*filters*spatial + f*spatial + i;
        float v = x[x_index];
        sum += v * v;
    }
    out[index] = sqrtf(sum) + eps;
}

extern "C"
__global__ void l2norm_1dim_kernel2_back(int N, float *x, float* delta, float *dx, int batch, int filters, int spatial, float eps)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int x_index = b*filters*spatial + f*spatial + i;
        float v = x[x_index];
        sum += v * v;
    }
    
    float tmp = delta[index] * (0.5 * powf(sum + eps, -0.5));
    
    for(f = 0; f < filters; ++f){
        int x_index = b*filters*spatial + f*spatial + i;
        float v = x[x_index];
        dx[x_index] = tmp * 2 * v;
    }

}