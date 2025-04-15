#define BLOCK 1024 


extern "C"
__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

extern "C"
__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

extern "C"
__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

extern "C"
__global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] += ALPHA;
}

extern "C"
__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

extern "C"
__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

extern "C"
__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

extern "C"
__global__ void scal_add_kernel(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

extern "C"
__global__ void constrain_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = fminf(ALPHA, fmaxf(-ALPHA, X[i*INCX]));
}

extern "C"
__global__ void concat_channel_forward_kernel(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        // copy input from x1
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;
        out[out_idx] = x1[idx];
    }
    if (idx < B * C2 * H * W) {
        // copy input from x2
        // move over from x1
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        out[out_idx] = x2[idx];
    }
}

extern "C"
__global__ void concat_channel_backward_kernel(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;

        dx1[idx] = dout[out_idx];
    }
    if (idx < B * C2 * H * W) {
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        dx2[idx] = dout[out_idx];
    }
}

extern "C"
__global__ void replace_channel_forward_kernel(
    float* out,
    const float* x1, const float* x2,
    int B, int C, int H, int W,int N, float* indices,int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int b = idx / (C * H * W);
        int c = (idx / H / W) % C;
        int h = (idx / W) % H;
        int w = idx % W;

		int startC = (int)indices[b];

		if(c >= startC && c < startC + size){
			int c2 = c - startC;
		    int out_idx = b * size * H * W + c2 * H * W + h * W + w;
			out[idx] = x2[out_idx];
		}else{
			out[idx] = x1[idx];
		}

    }
}

extern "C"
__global__ void replace_channel_backward_kernel(
    float* diff,
    float* dx,
    int B, int C, int H, int W,int N, float* indices,int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int b = idx / (C * H * W);
        int c = (idx / H / W) % C;
        int h = (idx / W) % H;
        int w = idx % W;

		int startC = (int)indices[b];

		if(c >= startC && c < startC + size){
			int c2 = c - startC;
		    int out_idx = b * size * H * W + c2 * H * W + h * W + w;
		    dx[out_idx] = diff[idx];
		    diff[idx] = 0.0f;
		}

    }
}

extern "C"
__global__ void add_mul_kernel(
    float* input,
    float* noise,
    float* output,
    float* a,
    float* b,
    int N, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       output[idx] = a[n] * input[idx] + noise[idx] * b[n];
    }
}

extern "C"
__global__ void un_mul_kernel(
    float* input,
    float* noise,
    float* output,
    float* a,
    float* b,
    int N, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       output[idx] = (input[idx] - noise[idx] * b[n]) / a[n];
    }
}

extern "C"
__global__ void un_mul_grad_kernel(
    float* delta,
    float* noise,
    float* diff,
    float* a,
    float* b,
    int N, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       diff[idx] = - delta[idx] / a[n] * b[n];
    }
}