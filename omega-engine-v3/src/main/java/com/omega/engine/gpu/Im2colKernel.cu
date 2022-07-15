extern "C"
__global__ void im2col_gpu(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int ow,int oh,int kSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
	int n = i / oHeight / oWidth;
			
	int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
	
	int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;

	for(int j = 0;j<ow;j++) {
		
		int c = j / kSize;
		
		int xSize = j - (c * kSize);
		
		int xh = startH + xSize / kw;
		
		int xw = startW + xSize % kw;
		
		int xi = n * C * H * W + c * H * W + xh * W + xw;
		
		out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];

	}

}

extern "C"
__global__ void im2col_gpuv2(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int ow,int oh,int kSize)
{
	
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int s = blockDim.x * gridDim.x;
    
    //printf("Thread %d %d\n", index, s);
   
	for(int i = index;i < oh;i += s){
		
		int n = i / oHeight / oWidth;
				
		int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
		
		int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
	
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
			
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kw;
			
			int xw = startW + xSize % kw;
			
			int xi = n * C * H * W + c * H * W + xh * W + xw;
			
			out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
		
	}
	
}

extern "C"
__global__ void test(int n,int *o)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

   	
}