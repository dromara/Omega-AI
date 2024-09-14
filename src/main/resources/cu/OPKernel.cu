#define BLOCK 1024
#define FLT_MAX 3.402823466e+38F

extern "C"
__global__ void fill_kernel(int N, float ALPHA, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i] = ALPHA;
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
__global__ void sum_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < 1) {
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
