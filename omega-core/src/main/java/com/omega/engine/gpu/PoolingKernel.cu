extern "C"
__global__ void max_pooling(float* x,float* mask,float* result,int n,int height,int width,int oHeight,int oWidth,int pWidth,int pHeight,int stride)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    
    	int c = index / (oHeight * oWidth);
    	
    	int oh = (index / oWidth) % oHeight;
    	
    	int ow = index % oWidth;
    	
    	int maskIndex = oh * oWidth + ow;
    	
    	int maxH = 0;
		int maxW = 0;
		
		float maxval = -3.402823466e+38;
		
		for(int m = 0;m<pHeight;m++) {	
			for(int n = 0;n<pWidth;n++) {
				float val = x[c * height * width + (oh * stride + m) * width + (ow * stride + n)];
				if(maxval <= val) {
					maxH = m;
					maxW = n;
					maxval = val;
				}
			}
		}	
		
		result[c * oHeight * oWidth + oh * oWidth + ow] = maxval;
		mask[c * oHeight * oWidth * pHeight * pWidth + maskIndex * pHeight * pWidth + maxH * pWidth + maxW] = 1;
	
	}

}


extern "C"
__global__ void mean_pooling(float* x,float* mask,float* result,int n,int height,int width,int oHeight,int oWidth,int pWidth,int pHeight,int stride)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    
    	int c = index / (oHeight * oWidth);
    	
    	int oh = (index / oWidth) % oHeight;
    	
    	int ow = index % oWidth;
    	
    	int maskIndex = oh * oWidth + ow;
    	
		float val = 0;
		
		for(int m = 0;m<pHeight;m++) {	
			for(int n = 0;n<pWidth;n++) {
				val += x[c * height * width + (oh * stride + m) * width + (ow * stride + n)];
				mask[c * oHeight * oWidth * pHeight * pWidth + maskIndex * pHeight * pWidth + m * pWidth + n] = 1.0f / (pWidth * pHeight);
			}
		}	
		
		result[c * oHeight * oWidth + oh * oWidth + ow] = val / (pWidth * pHeight);
		
	}

}

extern "C"
__global__ void pooling_diff(float* x,float* mask,float* diff,int n,int height,int width,int oHeight,int oWidth,int pWidth,int pHeight,int stride)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    
    	int c = index / (oHeight * oWidth);
    	
    	int oh = (index / oWidth) % oHeight;
    	
    	int ow = index % oWidth;
    	
    	int maskIndex = oh * oWidth + ow;
    	
    	float d_val = x[c * oHeight * oWidth + oh * oWidth + ow];
    	
		for(int m = 0;m<pHeight;m++) {	
			for(int n = 0;n<pWidth;n++) {
				diff[c * height * width + (oh * stride + m) * width + (ow * stride + n)] = d_val * mask[c * oHeight * oWidth * pHeight * pWidth + maskIndex * pHeight * pWidth + m * pWidth + n];
			}
		}	
		
	}

}