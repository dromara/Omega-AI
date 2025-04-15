#define BLOCK 1024 

extern "C"
__global__ void upsample_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

	if(forward) out[out_index] = scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
	//if(forward) out[out_index] = scale * x[in_index];
    //else x[in_index] += scale * out[out_index];
}
