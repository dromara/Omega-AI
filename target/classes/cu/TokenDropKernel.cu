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

extern "C"
__global__ void token_drop_class(const size_t size, const float *x, float param, float *mask, float *out, const int len, float prob)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / len;
    	if(mask[b] < prob){
			out[pos] = param;
		}else{
			out[pos] = x[pos];
		}
  	}

}

extern "C"
__global__ void timestep_embedding(int N, const float *t, float *freqs, float *out, const int d_model) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / (d_model / 2);
       int w = idx % (d_model / 2);
       float args = t[n] * freqs[w];
       out[n * d_model + w] = cos(args);
       out[n * d_model + (d_model / 2) + w] = sin(args);
    }
}
