extern "C"
__global__ void token_drop(const size_t size, const float *x, float *param, float *mask, float *out, const int len, float prob)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / len;
    	int w = pos % len;
    	if(mask[b] < prob){
			out[pos] = param[w];
		}else{
			out[pos] = x[pos];
		}
  	}

}
