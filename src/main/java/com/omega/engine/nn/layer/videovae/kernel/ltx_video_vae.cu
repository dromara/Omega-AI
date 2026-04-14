extern "C"
__global__ void encoder_repeat(size_t size, const float *x, float *out, const int C, const int F, const int H, const int W, int xIndex, int count) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
	int os = C * F * H * W;
	int ocs = F * H * W;
    int b = pos / os;
    int p_os = pos % os;
    int c = p_os / ocs;
    int p_ocs = p_os % ocs;
    int f = p_ocs / (H * W);
    int h = p_ocs % (H * W) / W;
    int w = p_ocs % (H * W) % W;
    int xc = c;
    if(c >= xIndex){
    	xc = xIndex;
    }
    int xos = (C - count) * ocs;
    int xidx = b * xos + xc * ocs + f * H * W + h * W + w;
    out[pos] = x[xidx];
  }
}

