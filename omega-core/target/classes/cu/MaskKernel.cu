extern "C"
__global__ void createHeadMask(const size_t size, const float *lens, float *mask,const int number,const int maxLen,const int headNum)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / headNum / maxLen / maxLen;
	    int pl = static_cast<int64_t>(lens[b]);
	    int w = pos % maxLen;
	    if(w < pl){
	    	mask[pos] = 0;
	    }else{
	    	mask[pos] = 1;
	    }
  	}

}
