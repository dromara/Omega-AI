extern "C"
__global__ void constPadding2d(const size_t size, const float *input, const int64_t num, const int64_t channels,
							  const int64_t old_height, const int64_t old_width,
                              const int64_t old_hw,
                              const int64_t padded_height, const int64_t padded_width,
                              const int64_t padded_hw, const int64_t pad_top,
                              const int64_t pad_left, const float pad_value, float *output)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
	    const int pos_h = pos / padded_width % padded_height;
	    const int pos_w = pos % padded_width;

	    if (pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
	      output[pos] = pad_value;
	    } else {
	      const int block_num = pos / padded_hw;
	      int index = block_num * old_hw + old_width * (pos_h - pad_top) + pos_w - pad_left;
	      output[pos] = input[index];
	    }
  	}

}

extern "C"
__global__ void ConstantPadGrad2d(const size_t size, const float *dy, const int64_t num, const int64_t channels,
                                  const int64_t old_height, const int64_t old_width,
                                  const int64_t old_hw,
                                  const int64_t padded_height, const int64_t padded_width,
                                  const int64_t padded_hw, const int64_t pad_top,
                                  const int64_t pad_left, float *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int block_num = pos / old_hw;
    const int pos_h = pos / old_width % old_height + pad_top;
    const int pos_w = pos % old_width + pad_left;
    const int index = block_num * padded_hw + pos_h * padded_width + pos_w;
    dx[pos] = dy[index];
  }
}

extern "C"
__global__ void constPadding3d(const size_t size, const float *input, const int64_t num, const int64_t channels,
                              const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                              const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                              const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                              const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, const float pad_value, float *output)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
	    const int pos_d = pos / padded_hw % padded_depth;
	    const int pos_h = pos / padded_width % padded_height;
	    const int pos_w = pos % padded_width;

	    if (pos_d - pad_head < 0 || pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_d - pad_head >= old_depth ||
	        pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
	      output[pos] = pad_value;
	    } else {
	      const int block_num = pos / padded_dhw;
	      int index = block_num * old_dhw + old_hw * (pos_d - pad_head) + old_width * (pos_h - pad_top) + pos_w - pad_left;
	      output[pos] = input[index];
	    }
  	}

}

extern "C"
__global__ void constPadding3d_seft(const size_t size, const float *input, const int64_t num, const int64_t channels,
                              const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                              const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                              const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                              const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, float *output)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
	    const int pos_d = pos / padded_hw % padded_depth;
	    const int pos_h = pos / padded_width % padded_height;
	    const int pos_w = pos % padded_width;
	    const int block_num = pos / padded_dhw;
	    if (pos_d - pad_head < 0 || pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_d - pad_head >= old_depth ||
	        pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
		  int oidx = pos - block_num * pad_head * old_hw;
	      output[pos] = input[oidx];
	    } else {
	      int index = block_num * old_dhw + old_hw * (pos_d - pad_head) + old_width * (pos_h - pad_top) + pos_w - pad_left;
	      output[pos] = input[index];
	    }
  	}

}

extern "C"
__global__ void ConstantPadGrad3d(const size_t size, const float *dy, const int64_t num, const int64_t channels,
                                  const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                  const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                                  const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                                  const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                                  const int64_t pad_left, float *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int block_num = pos / old_dhw;
    const int pos_d = pos / old_hw % old_depth + pad_head;
    const int pos_h = pos / old_width % old_height + pad_top;
    const int pos_w = pos % old_width + pad_left;
    const int index = block_num * padded_dhw + pos_d * padded_hw + pos_h * padded_width + pos_w;
    dx[pos] = dy[index];
  }
}

extern "C"
__global__ void ConstantPadGrad3d_self(const size_t size, const float *dy, const int64_t num, const int64_t channels,
                              const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                              const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                              const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                              const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, float *dx){

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
	    const int pos_d = pos / padded_hw % padded_depth;
	    const int pos_h = pos / padded_width % padded_height;
	    const int pos_w = pos % padded_width;
		const int block_num = pos / padded_dhw;
	    if (pos_d - pad_head < 0 || pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_d - pad_head >= old_depth ||
	        pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
		  int oidx = pos - block_num * pad_head * old_hw;		
	      dx[oidx] += dy[pos];
	    }
  	}

}
