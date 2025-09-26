#define BLOCK 1024
#define FLT_MAX 3.402823466e+38F


extern "C"
__global__ void fill_kernel(int N, float ALPHA, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i] = ALPHA;
}

extern "C"
__global__ void abs_kernel(int N, float* x, float *y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) y[i] = fabsf(x[i]);
}

extern "C"
__global__ void abs_backward_kernel(int N, float* x, float* dx, float *dy)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
		float dxv = x[i] == 0.0 ? 0.0 : copysignf(1.0, x[i]);
		dx[i] = dxv * dy[i];
	} 
}

extern "C"

__global__ void copy_kernel(int N,  float *X, int OFFX, float *Y, int OFFY)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	Y[i + OFFY] = X[i + OFFX];

    }

}



extern "C"

__global__ void axpy_kernel(int N,  float *X, int OFFX, float *Y, int OFFY)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	Y[i + OFFY] += X[i + OFFX];

    }

}



extern "C"

__global__ void one_hot_kernel(int N,  float *X, float *Y, int K)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int idx = (int)X[i];

    	Y[i * K + idx] = 1.0f;

    }

}



extern "C"

__global__ void copy_number_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start,int cp)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int size = c * h * w;

    	int tn = i / size + start;

		int tc = i / h / w % c;

		int th = i / w % h;

		int tw = i % w;

		int index = tn * size + tc * h * w + th * w + tw;

		if(cp == 0){

			Y[i] = X[index];

		}else{

			X[index] = Y[i];

		}

    }

}



extern "C"
__global__ void copy_channel_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start,int cp)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int bc = N / n / h / w;

		int size = bc * h * w;

    	int tn = i / size;

		int tc = (i / h / w) % bc + start;
		
		int th = i / w % h;

		int tw = i % w;

		int index = tn * c * h * w + tc * h * w + th * w + tw;

    	if(cp == 0){

			Y[i] = X[index];

		}else{

			X[index] = Y[i];

		}

    }

}

extern "C"
__global__ void get_by_channel_kenel(int N, float *X, float *Y, int C,int H,int W, float *ids)
{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){
		
		int n = i / H / W;
		int c = (int) ids[n];
		int hw = i % (H * W);
		
		int index = n * C * H * W + c * H * W + hw;

    	Y[i] = X[index];

    }

}

extern "C"

__global__ void broadcast_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = X[0];

}



extern "C"

__global__ void broadcast_number_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	int n = i / C / H / W;

    	Y[i] = X[n];

    }

}



extern "C"

__global__ void broadcast_row_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	int n = i % (C * H * W);

    	Y[i] = X[n];

    }

}



extern "C"

__global__ void broadcast_plus_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] += X[0];

}



extern "C"

__global__ void broadcast_number_plus_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	int n = i / C / H / W;

    	Y[i] += X[n];

    }

}



extern "C"

__global__ void broadcast_row_plus_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	int n = i % (C * H * W);

    	Y[i] += X[n];

    }

}



extern "C"

__global__ void add_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] + Y[i];

}



extern "C"

__global__ void add_axis_kernel(int N, float *X, float *Y, float *R,int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i / axis;

    	R[i] = X[i] + Y[yi];

    } 

}


extern "C"
__global__ void add_axis_kernel2(int N, float *X, float *Y, float *R,int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i % axis;

    	R[i] = X[i] + Y[yi];

    } 

}

extern "C"
__global__ void add_axis_kernel3(int N, float *X, float *Y, float *R, int B, int C, int H, int W, int axis)
{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = 0;
    	int len = C * H * W;
    	
    	if(axis == 0){
			yi = i % len;
		}else if(axis == 1){
			int n = i / len;
			yi = n * (H * W) + i % (H * W);
		}

    	R[i] = X[i] + Y[yi];

    } 

}

extern "C"
__global__ void add_axis_back_kernel(int N, float *dX, float *dY, int B, int C, int H, int W, int axis)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
		float dy = 0;
    	if(axis == 0){
			for(int j = 0;j<B;j++){
				dy += dY[j * C * H * W + i];
			}
		}else if(axis == 1){
			int n = i / H / W;
			int hw = i % (H * W);
			for(int j = 0;j<C;j++){
				dy += dY[n * C * H * W + j * H * W + hw];
			}
		}
    	dX[i] = dy;
    } 
}

extern "C"
__global__ void mul_axis_kernel(int N, float *X, float *Y, float *R, int B, int C, int H, int W, int axis)
{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = 0;
    	int len = C * H * W;
    	
    	if(axis == 0){
			yi = i % len;
		}else if(axis == 1){
			int n = i / len;
			yi = n * (H * W) + i % (H * W);
		}

    	R[i] = X[i] * Y[yi];

    } 

}

extern "C"
__global__ void mul_axis_kernel2(int N, float *X, float *Y, float *R, int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i % axis;

    	R[i] = X[i] * Y[yi];

    } 

}

extern "C"
__global__ void mul_axis_back_kernel(int N, float *X, float *Y, float *R, int WH)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
		R[i] = 0;
    	for(int wh = 0;wh<WH;wh++){
			R[i] += X[i * WH + wh] * Y[i * WH + wh];
		}
    } 
}

extern "C"
__global__ void mul_axis_back_left_kernel(int N, float *Y, float *delta, float *dx, int B, int C, int H, int W, int axis)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = 0;
    	int len = C * H * W;
    	
    	if(axis == 0){
			yi = i % len;
		}else if(axis == 1){
			int n = i / len;
			yi = n * (H * W) + i % (H * W);
		}

    	dx[i] = delta[i] * Y[yi];

    } 
}

extern "C"
__global__ void mul_axis_back_right_kernel(int N, float *X, float *delta, float *dY, int B, int C, int H, int W, int axis)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
		float dy = 0;
    	if(axis == 0){
			for(int j = 0;j<B;j++){
				dy += delta[j * C * H * W + i] * X[j * C * H * W + i];
			}
		}else if(axis == 1){
			int n = i / H / W;
			int hw = i % (H * W);
			for(int j = 0;j<C;j++){
				dy += delta[n * C * H * W + j * H * W + hw] * X[n * C * H * W + j * H * W + hw];
			}
		}
    	dY[i] = dy;
    }
}

extern "C"
__global__ void expand_kernel(int N, float *X, float *Y, int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int xi = i % axis;

    	Y[i] = X[xi];

    } 

}



extern "C"

__global__ void sum_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < 1) {
		
		Y[0] = 0;
		
	    for(int index = 0;index<N;index++){

	    	Y[0] += X[index];

	    }

    }

}

extern "C"
__global__ void sum_channel_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	for(int index = 0;index<C * H * W;index++){

    		Y[i] += X[i * C * H * W + index];

    	}

    }

}

extern "C"
__global__ void sum_height_kernel(int N, float *X, float *Y,int C,int H,int W)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	Y[i] = 0;

    	for(int index = 0;index<H * W;index++){

    		Y[i] += X[i * H * W + index];

    	}

    }

}



extern "C"

__global__ void sum_pow_kernel(int N, double p, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < 1) {

	    for(int index = 0;index<N;index++){

	    	Y[0] += powf(X[index], p);

	    }

    }

}



extern "C"

__global__ void sum_pow_channel_kernel(int N, double p, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	for(int index = 0;index<C * H * W;index++){

    		Y[i] += powf(X[i * C * H * W + index], p);

    	}

    }

}



extern "C"

__global__ void sum_pow_height_kernel(int N, double p, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	Y[i] = 0;

    	for(int index = 0;index<H * W;index++){

    		Y[i] += powf(X[i * H * W + index], p);

    	}

    }

}



extern "C"

__global__ void max_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < 1) {

    	float max = -FLT_MAX;

	    for(int index = 0;index<N;index++){

	    	if(max <= X[index]){

	    		max = X[index];

	    	}

	    }

	    Y[0] = max;

    }

}



extern "C"

__global__ void max_channel_kernel(int N, float *X, float *Y,int C,int H,int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	float max = -FLT_MAX;

    	for(int index = 0;index<C * H * W;index++){

    		if(max <= X[i * C * H * W + index]){

	    		max = X[i * C * H * W + index];

	    	}

    	}

    	Y[i] = max;

    }

}



extern "C"

__global__ void max_backward_kernel(int N, float *D, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < 1) {

    	float max = -FLT_MAX;

    	int max_idx = 0;

	    for(int index = 0;index<N;index++){

	    	if(max <= X[index]){

	    		max = X[index];

	    		max_idx = index;

	    	}

	    }

	    Y[max_idx] += D[0];

    }

}



extern "C"

__global__ void max_channel_backward_kernel(int N, float *D, float *X, float *Y, int C, int H, int W)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	float max = -FLT_MAX;

    	int max_idx = 0;

    	for(int index = 0;index<C * H * W;index++){

    		if(max <= X[i * C * H * W + index]){

	    		max = X[i * C * H * W + index];

	    		max_idx = i * C * H * W + index;

	    	}

    	}

    	Y[max_idx] += D[i];

    }

}



extern "C"

__global__ void add_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] + ALPHA;

}



extern "C"

__global__ void add_number_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int size = c * h * w;

    	int tn = i / size + start;

		int tc = (i / h / w) % c;

		int th = i / w % h;

		int tw = i % w;

		int index = tn * size + tc * h * w + th * w + tw;

    	X[index] += Y[i];

    }

}



extern "C"

__global__ void add_channel_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int bc = N / n / h / w;

		int size = bc * h * w;

    	int tn = i / size;

		int tc = (i / h / w) % bc + start;

		int th = i / w % h;

		int tw = i % w;

		int index = tn * c * h * w + tc * h * w + th * w + tw;

    	X[index] += Y[i];

    }

}



extern "C"

__global__ void sub_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] - Y[i];

}



extern "C"

__global__ void sub_axis_kernel(int N, float *X, float *Y, float *R,int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i / axis;

    	R[i] = X[i] - Y[yi];

    } 

}



extern "C"

__global__ void sub_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] - ALPHA;

}



extern "C"

__global__ void scalar_sub_kernel(int N, float ALPHA, float *X, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = ALPHA - X[i];

}



extern "C"
__global__ void bool_kernel(int N, float *X, float *Y, float *R,float val)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	if(Y[i] == 1){

    		R[i] = val;

    	}else{

    		R[i] = X[i];

    	}

    } 

}

extern "C"
__global__ void mask_kernel(int N, float *X, float *Y, float *R, int onceSize,float val)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){
		
		int idx = i % onceSize;
		
    	if(Y[idx] == 1){

    		R[i] = val;

    	}else{

    		R[i] = X[i];

    	}

    } 

}

extern "C"
__global__ void mul_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] * Y[i];

}



extern "C"

__global__ void mul_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] * ALPHA;

}



extern "C"

__global__ void mul_plus_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] += X[i] * Y[i];

}



extern "C"

__global__ void mul_plus_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] += X[i] * ALPHA;

}



extern "C"

__global__ void mul_plus_scalar_axis_kernel(int N, float *X, float ALPHA, float *R, int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	for(int xi = 0;xi<axis;xi++){

    		R[i] += X[i * axis + xi] * ALPHA;

    	}

    }

}



extern "C"

__global__ void div_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] / Y[i];

}



extern "C"

__global__ void div_axis_kernel(int N, float *X, float *Y, float *R,int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i / axis;

    	R[i] = X[i] / Y[yi];

    } 

}



extern "C"

__global__ void div_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = X[i] / ALPHA;

}



extern "C"

__global__ void scalar_div_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] = ALPHA / X[i];

}



extern "C"

__global__ void div_bGrad_kernel(int N, float *D, float *A, float *B, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] += - 1.0f * D[i] * A[i] / (B[i] * B[i]); 

}



extern "C"

__global__ void div_bGrad_axis_kernel(int N, float *D, float *A, float *B, float *Y,int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	for(int di = 0;di<axis;di++){

    		Y[i] += (- 1.0f * D[i * axis + di] * A[i * axis + di]) / (B[i] * B[i]); 

    	}

    } 

}



extern "C"

__global__ void div_scalar_bGrad_kernel(int N, float *D, float A, float *B, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] += - 1.0f * D[i] * A / (B[i] * B[i]); 

}



extern "C"

__global__ void div_plus_kernel(int N, float *X, float *Y, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] += X[i] / Y[i];

}



extern "C"

__global__ void div_plus_axis_kernel(int N, float *X, float *Y, float *R, int axis)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N){

    	int yi = i / axis;

    	R[i] += X[i] / Y[yi];

    } 

}



extern "C"

__global__ void div_plus_scalar_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] += X[i] / ALPHA;

}



extern "C"

__global__ void scalar_plus_div_kernel(int N, float *X, float ALPHA, float *R)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) R[i] += ALPHA / X[i];

}



extern "C"

__global__ void pow_kernel(int N, float *X, float ALPHA, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = powf(X[i], ALPHA);

}



extern "C"

__global__ void sqrt_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = sqrtf(X[i]);

}



extern "C"

__global__ void log_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	//if(X[i] == 0){

    		//X[i] = 0.00000000000000000000001f;

    	//}

   	 	Y[i] = logf(X[i]);

    }

}



extern "C"

__global__ void exp_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = expf(X[i]);

}



extern "C"

__global__ void sin_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = sin(X[i]);

}



extern "C"

__global__ void cos_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = cos(X[i]);

}



extern "C"

__global__ void tan_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = tan(X[i]);

}



extern "C"

__global__ void tan_back_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = 1 / powf(cos(X[i]), 2);

}



extern "C"

__global__ void atan_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = atan(X[i]);

}



extern "C"

__global__ void atan_back_kernel(int N, float *X, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) Y[i] = 1.0f / (1 + X[i] * X[i]);

}



extern "C"

__global__ void clamp_kernel(int N, float *X, float min, float max, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	float val = X[i];

		if(val < min) {

			Y[i] = min;

		}else if(val > max) {

			Y[i] = max;

		}else {

			Y[i] = val;

		}

    }

}



extern "C"

__global__ void clamp_back_kernel(int N, float *X, float min, float max, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

    	float val = X[i];

		if(val < min || val > max) {

			Y[i] = 0;

		}else {

			Y[i] = 1;

		}

    }

}



extern "C"

__global__ void maximum_kernel(int N, float *X, float *Z, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

		if(X[i] >= Z[i]) {

			Y[i] = X[i];

		}else {

			Y[i] = Z[i];

		}

    }

}



extern "C"

__global__ void minimum_kernel(int N, float *X, float *Z, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

		if(X[i] < Z[i]) {

			Y[i] = X[i];

		}else {

			Y[i] = Z[i];

		}

    }

}



extern "C"

__global__ void maximum_back_kernel(int N, float *X, float *Z, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

		if(X[i] >= Z[i]) {

			Y[i] = 1;

		}else {

			Y[i] = 0;

		}

    }

}



extern "C"

__global__ void minimum_back_kernel(int N, float *X, float *Z, float *Y)

{

    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < N) {

		if(X[i] < Z[i]) {

			Y[i] = 1;

		}else {

			Y[i] = 0;

		}

    }

}



extern "C"

__global__ void transpose_kernel(int N, float *A, float *B,int m,int n)

{	

	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

	int r = i / n;

	int c = i % n;

    if (i < N)

    {

        B[c * m + r] = A[r * n + c];

    }

}



extern "C"

__global__ void permute_kernel(int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES) {

    //int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {

        int offset_out = tid;

        int offset_tmp = offset_out;

        int offset_in = 0;

        for (int i = 0; i < NUM_AXES; i++) {

            offset_in += (offset_tmp / strides_out[i]) * strides_in[perms[i]];

            offset_tmp %= strides_out[i];

        }

        data_out[offset_out] = data_in[offset_in];

    }

}



extern "C"

__global__ void permute_add_kernel(int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES) {

    //int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {

        int offset_out = tid;

        int offset_tmp = offset_out;

        int offset_in = 0;

        for (int i = 0; i < NUM_AXES; i++) {

            offset_in += (offset_tmp / strides_out[i]) * strides_in[perms[i]];

            offset_tmp %= strides_out[i];

        }

        data_out[offset_out] += data_in[offset_in];

    }

}


extern "C"
__global__ void mean_kernel(int N, float *x, float *y, int C) {

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {

    	float sum = 0.0f;

        for (int i = 0; i < C; i++) {

            sum += x[tid * C + i];

        }

        y[tid] += sum / C;

    }

}

extern "C"
__global__ void mean_back_kernel(int N, float *dy, float *dx, int C) {

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {

        for (int i = 0; i < C; i++) {

            dx[tid * C + i] = dy[tid] / C;

        }

    }

}

extern "C"
__global__ void cat_width_kernel(int N, float *a, float *b, float *y, int W) {

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {
        for (int i = 0; i < W * 2; i++) {
			int oi = i / 2;
			if(i / W < 1){
				y[tid * W * 2 + i] = a[tid * W + oi];
			}else{
				y[tid * W * 2 + i] = b[tid * W + oi];
			}
        }
    }

}

extern "C"
__global__ void cat_width_back_kernel(int N, float *da, float *db, float *dy, int W) {

    int tid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (tid < N) {
        for (int i = 0; i < W * 2; i++) {
			int oi = i / 2;
			if(i / W < 1){
				da[tid * W + oi] = dy[tid * W * 2 + i];
			}else{
				db[tid * W + oi] = dy[tid * W * 2 + i];
			}
        }
    }

}


extern "C"
__global__ void update_ema(int N,float *ema, float *model,float decay)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
	ema[i] = ema[i] * decay + model[i] * (1 - decay);
}