extern "C"
__global__ void im2col_gpu_kernel(float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int oh,int ow) {

    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
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
__global__ void im2col_gpu_kernelV2(const float* data_im,float* data_col,const int n,const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride,const int pad,const int height_col, const int width_col) {
  for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride - pad;
    const int w_offset = w_col * stride - pad;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
