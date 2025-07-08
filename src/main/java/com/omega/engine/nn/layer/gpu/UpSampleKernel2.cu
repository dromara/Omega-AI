#define BLOCK 1024 

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__device__ int translate_idx_inv(
    int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

extern "C"
__global__ void upscale(const float *input, float *output, int no_elements,
                        int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

extern "C"
__global__ void downscale(float *gradInput_data, const float *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}

__device__ int translate_idx_3d(int ii, int d1, int d2, int d3, int d4, int scale_factor) {
  int n, c, d, h, w;
  w = ii % d4;
  ii = ii/d4;
  h = ii % d3;
  ii = ii/d3;
  d = ii % d2;
  ii = ii/d2;
  c = ii % d1;
  ii = ii/d1;
  n = ii;
  d = d/scale_factor;
  h = h/scale_factor;
  w = w/scale_factor;

  d2 /= scale_factor;
  d3 /= scale_factor;
  d4 /= scale_factor;

  return (((n*d1+c)*d2+d)*d3+h)*d4+w;
}

extern "C"
__global__ void upscale3d(const float *input, float *output, int no_elements,
                        int scale_factor, int d1, int d2, int d3, int d4) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx_3d(ii, d1, d2, d3, d4, scale_factor);
  output[ii]=input[ipidx];
}

__device__ int translate_idx_3d_inv(
    int ii, int d1, int d2, int d3, int d4, int scale_factor, int off_d, int off_x, int off_y) {
  int n, c, d, h, w;
  w = ii % d4;
  ii = ii/d4;
  h = ii % d3;
  ii = ii/d3;
  d = ii % d2;
  ii = ii/d2;
  c = ii % d1;
  ii = ii/d1;
  n = ii;
  d = d*scale_factor+off_d;
  w = w*scale_factor+off_x;
  h = h*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  d4 *= scale_factor;
  return (((n*d1+c)*d2+d)*d3+h)*d4+w;
}

extern "C"
__global__ void downscale3d(float *gradInput_data, const float *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2, int d3, int d4) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int d=0; d < scale_factor; d++){
	  for (int i=0; i < scale_factor; i++){
	    for(int j=0; j < scale_factor; j++){
	      int ipidx = translate_idx_3d_inv(ii, d1, d2, d3, d4, scale_factor, d, i, j);
	      gradInput_data[ii] += gradOutput_data[ipidx];
	    }
	  }
  }
}

__device__ __forceinline__ static float area_pixel_compute_source_index(float scale, int dst_idx, bool align_corners,
                                                                       bool cubic) {
  if (align_corners) {
    return scale * dst_idx;
  } else {
    float src_idx = scale * (dst_idx + static_cast<float>(0.5)) - static_cast<float>(0.5);
    return (!cubic && src_idx < static_cast<float>(0)) ? static_cast<float>(0) : src_idx;
  }
}

extern "C"
__global__ void UpsampleTrilinear3DKernel(const int num_kernels, float *input, float *output, const int batch_size,
                                          const int channel, const int in_d, const int in_h, const int in_w,
                                          const int out_d, const int out_h, const int out_w, float d_scale,
                                          float h_scale, float w_scale, const bool align_corners, const int in_dhw,
                                          const int out_hw, const int out_dhw) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    const int w2 = (pos % out_hw) % out_w;
    const int h2 = (pos % out_hw) / out_w;
    const int d2 = pos / out_hw;
    // calculate scaled values for input index
    
    const float t1r = area_pixel_compute_source_index(d_scale, d2, align_corners, false);
    const int t1 = t1r;
    const int t1p = (t1 < in_d - 1) ? t1 + 1 : t1;
    const float lambda_d1 = t1r - t1;
    const float lambda_d0 = static_cast<float>(1) - lambda_d1;
    //
    const float h1r = area_pixel_compute_source_index(h_scale, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < in_h - 1) ? h1 + 1 : h1;
    const float lambda_h1 = h1r - h1;
    const float lambda_h0 = static_cast<float>(1) - lambda_h1;
    //
    const float w1r = area_pixel_compute_source_index(w_scale, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < in_w - 1) ? w1 + 1 : w1;
    const float lambda_w1 = w1r - w1;
    const float lambda_w0 = static_cast<float>(1) - lambda_w1;
    //
    auto in_data = input;
    auto out_data = output;
    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channel; ++c) {
        const float val = lambda_d0 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1p]))) +
                      lambda_d1 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1p])));
        out_data[(d2 * out_h + h2) * out_w + w2] = static_cast<float>(val);
        in_data += in_dhw;
        out_data += out_dhw;
      }
    }
  }
  return;
}

extern "C"
__global__ void UpsampleTrilinear3DKernel_igone(const int num_kernels, float *input, float *output, const int batch_size,
                                          const int channel, const int in_d, const int in_h, const int in_w,
                                          const int out_d, const int out_h, const int out_w, float d_scale,
                                          float h_scale, float w_scale, const bool align_corners, const int in_dhw,
                                          const int out_hw, const int out_dhw, const int igoneIndex) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    const int w2 = (pos % out_hw) % out_w;
    const int h2 = (pos % out_hw) / out_w;
    const int d2 = pos / out_hw;
    // calculate scaled values for input index
    if(d2 < igoneIndex){
		d_scale = 1;
	}else if(d2 == igoneIndex){
		return;
	}
    const float t1r = area_pixel_compute_source_index(d_scale, d2, align_corners, false);

    const int t1 = t1r;
    const int t1p = (t1 < in_d - 1) ? t1 + 1 : t1;
    const float lambda_d1 = t1r - t1;
    const float lambda_d0 = static_cast<float>(1) - lambda_d1;
    //
    const float h1r = area_pixel_compute_source_index(h_scale, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < in_h - 1) ? h1 + 1 : h1;
    const float lambda_h1 = h1r - h1;
    const float lambda_h0 = static_cast<float>(1) - lambda_h1;
    //
    const float w1r = area_pixel_compute_source_index(w_scale, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < in_w - 1) ? w1 + 1 : w1;
    const float lambda_w1 = w1r - w1;
    const float lambda_w0 = static_cast<float>(1) - lambda_w1;
    //
    auto in_data = input;
    auto out_data = output;
    
    int td2 = d2;
    
    if(d2 > igoneIndex){
		td2 = d2 - 1;
	}
    
    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channel; ++c) {

        const float val = lambda_d0 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1p]))) +
                      lambda_d1 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1p])));
        out_data[(td2 * out_h + h2) * out_w + w2] = static_cast<float>(val);
        in_data += in_dhw;
        out_data += (out_d - 1) * out_hw;
      }
    }
  }
  return;
}

extern "C"
__global__ void UpsampleTrilinear3DKernel_offset(const int num_kernels, float *input, float *output, const int batch_size,
                                          const int channel, const int in_d, const int in_h, const int in_w,
                                          const int out_d, const int out_h, const int out_w, float d_scale,
                                          float h_scale, float w_scale, const bool align_corners, const int in_dhw,
                                          const int out_hw, const int out_dhw, const int offset) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    const int w2 = (pos % out_hw) % out_w;
    const int h2 = (pos % out_hw) / out_w;
    const int d2 = pos / out_hw;
    // calculate scaled values for input index
    
    const float t1r = area_pixel_compute_source_index(d_scale, d2, align_corners, false);
    const int t1 = t1r;
    const int t1p = (t1 < in_d - 1) ? t1 + 1 : t1;
    const float lambda_d1 = t1r - t1;
    const float lambda_d0 = static_cast<float>(1) - lambda_d1;
    //
    const float h1r = area_pixel_compute_source_index(h_scale, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < in_h - 1) ? h1 + 1 : h1;
    const float lambda_h1 = h1r - h1;
    const float lambda_h0 = static_cast<float>(1) - lambda_h1;
    //
    const float w1r = area_pixel_compute_source_index(w_scale, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < in_w - 1) ? w1 + 1 : w1;
    const float lambda_w1 = w1r - w1;
    const float lambda_w0 = static_cast<float>(1) - lambda_w1;
    //
    auto in_data = input;
    auto out_data = output;
    
    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channel; ++c) {
		in_data += offset * in_h * in_w;
        const float val = lambda_d0 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1 * in_h + h1p) * in_w + w1p]))) +
                      lambda_d1 * (lambda_h0 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<float>(in_data[(t1p * in_h + h1p) * in_w + w1p])));
        out_data[((d2 + offset) * out_h + h2) * out_w + w2] = static_cast<float>(val);
        in_data += (in_dhw - (offset * in_h * in_w));
        out_data += out_dhw;
      }
    }
  }
  return;
}

__device__ __forceinline__ float MsAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ void FastAtomicAdd(float *base, size_t offset, const size_t length, float value) {
  MsAtomicAdd(base + offset, value);
}

__device__ __forceinline__ int idx_dhw(const int height, const int width, const int z, const int y, const int x) {
  return (z * height + y) * width + x;
}

extern "C"
__global__ void UpsampleTrilinear3DGradKernel(const size_t elem_num, const float *grad, const int batchsize,
                                              const int channels, const int grad_d, const int grad_h, const int grad_w,
                                              const int grad_dhw, const int dinput_d, const int dinput_h,
                                              const int dinput_w, const int dinput_dhw, const float d_scale,
                                              const float h_scale, const float w_scale, const bool align_corner, float *dinput) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < grad_dhw; pos += blockDim.x * gridDim.x) {
    const int t2 = pos / (grad_h * grad_w);
    const int h2 = pos / grad_w % grad_h;
    const int w2 = pos % grad_w;

    const float t1r = area_pixel_compute_source_index(d_scale, t2, align_corner, false);
    const int t1 = floorf(t1r);
    const int t1p = (t1 < (dinput_d - 1)) ? 1 : 0;
    const float t1lambda = t1r - t1;
    const float t0lambda = static_cast<float>(1) - t1lambda;

    const float h1r = area_pixel_compute_source_index(h_scale, h2, align_corner, false);
    const int h1 = floorf(h1r);
    const int h1p = (h1 < (dinput_h - 1)) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = static_cast<float>(1) - h1lambda;

    const float w1r = area_pixel_compute_source_index(w_scale, w2, align_corner, false);
    const int w1 = floorf(w1r);
    const int w1p = (w1 < (dinput_w - 1)) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = static_cast<float>(1) - w1lambda;

    size_t dinput_offset = 0;
    size_t dout_offset = 0;
    for (int n = 0; n < batchsize; ++n) {
      for (int c = 0; c < channels; ++c) {
		
        const float d2val = grad[dout_offset + (t2 * grad_h + h2) * grad_w + w2];
		
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1), elem_num,
                      t0lambda * h0lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1 + w1p), elem_num,
                      t0lambda * h0lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1), elem_num,
                      t0lambda * h1lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1 + w1p), elem_num,
                      t0lambda * h1lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1), elem_num,
                      t1lambda * h0lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1 + w1p), elem_num,
                      t1lambda * h0lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1), elem_num,
                      t1lambda * h1lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1 + w1p), elem_num,
                      t1lambda * h1lambda * w1lambda * d2val);

        dout_offset += grad_dhw;
        dinput_offset += dinput_dhw;
      }
    }
  }
  return;
}

extern "C"
__global__ void UpsampleTrilinear3DGradKernel_offset(const size_t elem_num, const float *grad, const int batchsize,
                                              const int channels, const int grad_d, const int grad_h, const int grad_w,
                                              const int grad_dhw, const int tar_grad_dhw, const int dinput_d, const int dinput_h,
                                              const int dinput_w, const int dinput_dhw, const float d_scale,
                                              const float h_scale, const float w_scale, const bool align_corner, float *dinput, const int offset) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < tar_grad_dhw; pos += blockDim.x * gridDim.x) {

    const int t2 = pos / (grad_h * grad_w);
    const int h2 = pos / grad_w % grad_h;
    const int w2 = pos % grad_w;

    const float t1r = area_pixel_compute_source_index(d_scale, t2, align_corner, false);
    const int t1 = floorf(t1r);
    const int t1p = (t1 < (dinput_d - 1)) ? 1 : 0;
    const float t1lambda = t1r - t1;
    const float t0lambda = static_cast<float>(1) - t1lambda;

    const float h1r = area_pixel_compute_source_index(h_scale, h2, align_corner, false);
    const int h1 = floorf(h1r);
    const int h1p = (h1 < (dinput_h - 1)) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = static_cast<float>(1) - h1lambda;

    const float w1r = area_pixel_compute_source_index(w_scale, w2, align_corner, false);
    const int w1 = floorf(w1r);
    const int w1p = (w1 < (dinput_w - 1)) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = static_cast<float>(1) - w1lambda;
	
    size_t dinput_offset = 0;
    size_t dout_offset = 0;
    for (int n = 0; n < batchsize; ++n) {
      for (int c = 0; c < channels; ++c) {
		
		dinput_offset += offset * dinput_h * dinput_w;
		
        const float d2val = grad[dout_offset + ((t2 + offset) * grad_h + h2) * grad_w + w2];

        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1), elem_num,
                      t0lambda * h0lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1 + w1p), elem_num,
                      t0lambda * h0lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1), elem_num,
                      t0lambda * h1lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1 + w1p), elem_num,
                      t0lambda * h1lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1), elem_num,
                      t1lambda * h0lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1 + w1p), elem_num,
                      t1lambda * h0lambda * w1lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1), elem_num,
                      t1lambda * h1lambda * w0lambda * d2val);
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1 + w1p), elem_num,
                      t1lambda * h1lambda * w1lambda * d2val);

        dout_offset += grad_dhw;
        dinput_offset += dinput_dhw - (offset * dinput_h * dinput_w);
      }
    }
  }
  return;
}