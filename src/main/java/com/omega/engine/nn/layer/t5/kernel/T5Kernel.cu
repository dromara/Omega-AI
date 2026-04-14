
extern "C"
__global__ void compute_bias(const size_t size, const float *x, float *mask, float *out, const int ht, const int len)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / len / ht;
    	int t = pos % (ht * len) / len;
    	int w = pos % len;
    	out[pos] = x[t * len + w] + mask[b * len + w];
  	}

}
