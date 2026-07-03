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
__global__ void concat_height_forward_kernel(
    const float* x1, const float* x2,
    float* out,
    int B, int C, int H1, int H2, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C * H1 * W) {
        // copy input from x1
        int b = idx / (C * H1 * W);
        int c = (idx / H1 / W) % C;
        int h = (idx / W) % H1;
        int w = idx % W;
        int out_idx = b * C * (H1 + H2) * W + c * (H1 + H2) * W + h * W + w;
        out[out_idx] = x1[idx];
    }
    if (idx < B * C * H2 * W) {
        // copy input from x2
        // move over from x1
        int b = idx / (C * H2 * W);
        int c = (idx / H2 / W) % C;
        int h = (idx / W) % H2;
        int w = idx % W;
        int out_idx = b * C * (H1 + H2) * W + c * (H1 + H2) * W + (H1 + h) * W + w;
        out[out_idx] = x2[idx];
    }
}

extern "C"
__global__ void concat_height_backward_kernel(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C, int H1, int H2, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C * H1 * W) {
        int b = idx / (C * H1 * W);
        int c = (idx / H1 / W) % C;
        int h = (idx / W) % H1;
        int w = idx % W;
        int out_idx = b * C * (H1 + H2) * W + c * (H1 + H2) * W + h * W + w;
        dx1[idx] = dout[out_idx];
    }
    if (idx < B * C * H2 * W) {
        int b = idx / (C * H2 * W);
        int c = (idx / H2 / W) % C;
        int h = (idx / W) % H2;
        int w = idx % W;
        int out_idx = b * C * (H1 + H2) * W + c * (H1 + H2) * (H1 + h) + h * W + w;
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

extern "C"
__global__ void cat_4d_dynamic_dim(
    const float* x,
    const float* y,
    float* out,
    int Bx, int Hx, int Nx, int Dx,
    int By, int Hy, int Ny, int Dy,
    int dim
) {
    int outB = (dim == 0) ? Bx + By : Bx;
    int outH = (dim == 1) ? Hx + Hy : Hx;
    int outN = (dim == 2) ? Nx + Ny : Nx;
    int outD = (dim == 3) ? Dx + Dy : Dx;

    int total = outB * outH * outN * outD;

    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % outD;
    int tmp = idx / outD;

    int n = tmp % outN;
    tmp /= outN;

    int h = tmp % outH;
    int b = tmp / outH;

    bool fromX = true;

    int xb = b;
    int xh = h;
    int xn = n;
    int xd = d;

    int yb = b;
    int yh = h;
    int yn = n;
    int yd = d;

    if (dim == 0) {
        if (b < Bx) {
            fromX = true;
        } else {
            fromX = false;
            yb = b - Bx;
        }
    } else if (dim == 1) {
        if (h < Hx) {
            fromX = true;
        } else {
            fromX = false;
            yh = h - Hx;
        }
    } else if (dim == 2) {
        if (n < Nx) {
            fromX = true;
        } else {
            fromX = false;
            yn = n - Nx;
        }
    } else { // dim == 3
        if (d < Dx) {
            fromX = true;
        } else {
            fromX = false;
            yd = d - Dx;
        }
    }

    if (fromX) {
        int xIdx = ((xb * Hx + xh) * Nx + xn) * Dx + xd;
        out[idx] = x[xIdx];
    } else {
        int yIdx = ((yb * Hy + yh) * Ny + yn) * Dy + yd;
        out[idx] = y[yIdx];
    }
}

extern "C"
__global__ void cat_4d_dynamic_dim_backward(
    const float* delta, // grad of cat output
    float* dx,
    float* dy,
    int Bx, int Hx, int Nx, int Dx,
    int By, int Hy, int Ny, int Dy,
    int dim
) {
    int outB = (dim == 0) ? Bx + By : Bx;
    int outH = (dim == 1) ? Hx + Hy : Hx;
    int outN = (dim == 2) ? Nx + Ny : Nx;
    int outD = (dim == 3) ? Dx + Dy : Dx;

    int total = outB * outH * outN * outD;

    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % outD;
    int tmp = idx / outD;

    int n = tmp % outN;
    tmp /= outN;

    int h = tmp % outH;
    int b = tmp / outH;

    if (dim == 0) {
        if (b < Bx) {
            int xIdx = ((b * Hx + h) * Nx + n) * Dx + d;
            dx[xIdx] = delta[idx];
        } else {
            int yb = b - Bx;
            int yIdx = ((yb * Hy + h) * Ny + n) * Dy + d;
            dy[yIdx] = delta[idx];
        }
    } else if (dim == 1) {
        if (h < Hx) {
            int xIdx = ((b * Hx + h) * Nx + n) * Dx + d;
            dx[xIdx] = delta[idx];
        } else {
            int yh = h - Hx;
            int yIdx = ((b * Hy + yh) * Ny + n) * Dy + d;
            dy[yIdx] = delta[idx];
        }
    } else if (dim == 2) {
        if (n < Nx) {
            int xIdx = ((b * Hx + h) * Nx + n) * Dx + d;
            dx[xIdx] = delta[idx];
        } else {
            int yn = n - Nx;
            int yIdx = ((b * Hy + h) * Ny + yn) * Dy + d;
            dy[yIdx] = delta[idx];
        }
    } else { // dim == 3
        if (d < Dx) {
            int xIdx = ((b * Hx + h) * Nx + n) * Dx + d;
            dx[xIdx] = delta[idx];
        } else {
            int yd = d - Dx;
            int yIdx = ((b * Hy + h) * Ny + n) * Dy + yd;
            dy[yIdx] = delta[idx];
        }
    }
}