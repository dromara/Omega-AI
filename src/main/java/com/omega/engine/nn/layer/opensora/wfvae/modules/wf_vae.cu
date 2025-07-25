extern "C"
__global__ void cat_number_expend(const size_t size, const float *x0, const float *x1,const float *x2, const float *x3,const float *x4, 
							const float *x5,const float *x6, const float *x7, int len, float *output)
{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int N = i / len;
    int w = i % len;
    int count = 8;
    if(i < size) {
		output[N * count * len + 0 * len + w] = x0[N * len + w];
		output[N * count * len + 1 * len + w] = x1[N * len + w];
		output[N * count * len + 2 * len + w] = x2[N * len + w];
		output[N * count * len + 3 * len + w] = x3[N * len + w];
		output[N * count * len + 4 * len + w] = x4[N * len + w];
		output[N * count * len + 5 * len + w] = x5[N * len + w];
		output[N * count * len + 6 * len + w] = x6[N * len + w];
		output[N * count * len + 7 * len + w] = x7[N * len + w];
	}

}

extern "C"
__global__ void cat_number_4_expend(const size_t size, const float *x0, const float *x1,const float *x2, const float *x3, int len, float *output)
{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int N = i / len;
    int w = i % len;
    int count = 4;
    if(i < size) {
		output[N * count * len + 0 * len + w] = x0[N * len + w];
		output[N * count * len + 1 * len + w] = x1[N * len + w];
		output[N * count * len + 2 * len + w] = x2[N * len + w];
		output[N * count * len + 3 * len + w] = x3[N * len + w];
	}
}

extern "C"
__global__ void append(int size, const float *x0, float *output, int len, int count, int currentIndex)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    int N = i / len;
    int w = i % len;
    if(i < size) {
        output[N * count * len + currentIndex * len + w] = x0[N * len + w];
    }
}

extern "C"
__global__ void cat_number(int size, const float **x, int count, int len, float *output)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int N = index / len;
    int w = index % len;
    if(index < size) {
		for(int i = 0;i<count;i++){
			output[N * count * len + i * len + w] = x[i][N * len + w];
		}
	}

}

extern "C"
__global__ void add_offset(int size, const float *x, float *output, int detph, int hw, int offset)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < size) {
	    int N = i / detph / hw;
	    int w = i % (detph * hw);
	    int d = w / hw + offset;
	    int idx = w % hw;
        output[i] += x[N * (detph + 1) * hw + d * hw + idx];
    }
}

