extern "C"
__global__ void im2col_gpu(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < oh){
   		
		int n = i / oHeight / oWidth;
				
		int startH = (i - (n * oHeight * oWidth)) / oWidth * stride;
		
		int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
	
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
				
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
   		
    }
   
}

extern "C"
__global__ void im2col_gpuv2(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
	
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int s = blockDim.x * gridDim.x;
    
    //printf("Thread %d %d\n", index, s);
   
	for(int i = index;i < oh;i += s){
		
		int n = i / oHeight / oWidth;
				
		int startH = (i - (n * oHeight * oWidth)) / oWidth * stride;
		
		int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
	
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
			
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
		
	}
	
}

extern "C"
__global__ void im2col_gpuv3(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
	
    int ih = blockIdx.x * blockDim.x + threadIdx.x;
    
    int iw = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ih < oh){
    	
	    int n = ih / oHeight / oWidth;
					
		int startH = (ih - (n * oHeight * oWidth)) / oWidth * stride;
			
		int startW = (ih - (n * oHeight * oWidth)) % oWidth * stride;
    
    	if(iw < ow){
    	
			int c = iw / kSize;
			
			int xSize = iw - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[ih * ow + iw] = x[n * C * H * W + c * H * W + xh * W + xw];
			
    	}
    }
    
}


extern "C"
__global__ void im2col_gpuv4(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
 
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    
     int ih = i / ow;
    
     int iw = i % ow;
    
	 if(ih < oh){
	     
	     int n = ih / oHeight / oWidth;
	     
	     int startH = (ih - (n * oHeight * oWidth)) / oWidth * stride;
	   
	  	 int startW = (ih - (n * oHeight * oWidth)) % oWidth * stride;
	    
	     if(iw < ow){
	     
	   		int c = iw / kSize;
	   
	   		int xSize = iw - (c * kSize);
	   
	   		int xh = startH + xSize / kh;
	   
	   		int xw = startW + xSize % kw;
	   
	   		out[ih * ow + iw] = x[n * C * H * W + c * H * W + xh * W + xw];
	   
	     }
	  }
    
}

extern "C"
__global__ void im2col_gpuV5(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
   
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < oh; i += blockDim.x * gridDim.x) {
   		
		int n = i / oHeight / oWidth;
				
		int startH = (i - (n * oHeight * oWidth)) / oWidth * stride;
		
		int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
	
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
				
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
   		
    }
   
}

extern "C"
__global__ void im2col_gpuV6(float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int oh,int ow,int kSize)
{
   
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int os = blockDim.x * gridDim.x;
   
   for (int i = index; i < oh; i += os) {
   
	 int startH = i / oWidth * stride;
		
	 int startW = i % oWidth * stride;
		
   	 for(int n = 0;n<N;n++){
   	 
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
				
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[n * oh * ow + i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
   		
     }
   
   }
   
}

extern "C"
__global__ void im2col_gpu_kernel(float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int oh,int ow) {

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int h_index = index / ow;
        const int h_col = h_index % oh;
        const int w_col = index % ow;
        const int c_im = h_index / oh;
        const int c_col = c_im * kh * kw;
        const int h_offset = h_col * s;
        const int w_offset = w_col * s;
        float* data_col_ptr = data_col;
        data_col_ptr += (c_col * oh + h_col) * ow + w_col;
        const float* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height&& w_im < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += oh * ow;
            }
        }
    }

}


extern "C"
__global__ void test(int n,int *o)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	
}