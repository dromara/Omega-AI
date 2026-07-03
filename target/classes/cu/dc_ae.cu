#define BLOCK 1024 

extern "C"
__global__ void channel_average(float *x, float *output, int C,int HxW,int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
		int b = i / C / HxW;
		int c = i / HxW % C;
		int hw = i % HxW;
		int ic = (c + C);
		int idx1 = b * C * 2 * HxW + c * HxW + hw;
		int idx2 = b * C * 2 * HxW + ic * HxW + hw;
    	output[i] = (x[idx1] + x[idx2]) / 2.0f;
    }
}

extern "C"
__global__ void channel_average_backward(float *dy, float *diff, int C, int HxW, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
		int b = i / C / HxW;
		int c = i / HxW % C;
		int hw = i % HxW;
		int ic = (c + C);
		int idx1 = b * C * 2 * HxW + c * HxW + hw;
		int idx2 = b * C * 2 * HxW + ic * HxW + hw;
		float v = dy[i] / 2;
		diff[idx1] = v;
		diff[idx2] = v;
    }
}

extern "C"
__global__ void channel_duplicate(float *x, float *output, int C, int HxW, int repeats, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
		int xC = C / repeats;
		int b = i / C / HxW;
		int c = i / HxW % C;
		int hw = i % HxW;
		int idx = b * xC * HxW + (c % xC) * HxW + hw;
    	output[i] = x[idx];
    }
}

extern "C"
__global__ void channel_duplicate_backward(float *dy, float *diff, int C, int HxW, int repeats, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
		float v = 0;
		int b = i / C / HxW;
		int c = i / HxW % C;
		int hw = i % HxW;
		for(int r = 0;r<repeats;r++){
			v += dy[b * C * repeats * HxW + (c + C * r) * HxW + hw];
		}
		diff[i] = v;
    }
}
