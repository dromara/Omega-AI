#define BLOCK 1024 

__device__ inline uint start_index(uint a, uint b, uint c) {
  return floorf(__uint2float_rn(a * c) / __uint2float_rn(b));
}

__device__ inline uint end_index(uint a, uint b, uint c) {
  return ceilf(__uint2float_rn((a + 1) * c) / __uint2float_rn(b));
}

extern "C"
__global__ void AdaptiveAvgPool2DKernel(const uint size, const uint input_height, const uint input_width,
                                        const uint output_height, const uint output_width, float *input_data,
                                        float *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    float *input_ptr = input_data + c * input_height * input_width;
    float *output_ptr = output_data + c * output_height * output_width;

    for (uint oh = 0; oh < output_height; oh++) {
      uint ih0 = start_index(oh, output_height, input_height);
      uint ih1 = end_index(oh, output_height, input_height);
      uint kh = ih1 - ih0;

      for (uint ow = 0; ow < output_width; ow++) {
        uint iw0 = start_index(ow, output_width, input_width);
        uint iw1 = end_index(ow, output_width, input_width);
        uint kw = iw1 - iw0;

        // compute local average
        float sum = 0;
        for (uint ih = ih0; ih < ih1; ih++) {
          for (uint iw = iw0; iw < iw1; iw++) {
            sum += input_ptr[ih * input_width + iw];
          }
        }
        output_ptr[oh * output_width + ow] = sum / __uint2float_rn(kh * kw);
      }
    }
  }
}

extern "C"
__global__ void AdaptiveAvgPool2DGradKernel(const uint size, const uint input_height, const uint input_width,
                                            const uint output_height, const uint output_width, float *input_data,
                                            float *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    float *input_ptr = input_data + c * input_height * input_width;
    float *output_ptr = output_data + c * output_height * output_width;

    size_t output_size = output_height * output_width;
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = 0;
    }

    for (uint ih = 0; ih < input_height; ih++) {
      uint oh0 = start_index(ih, input_height, output_height);
      uint oh1 = end_index(ih, input_height, output_height);
      uint kh = oh1 - oh0;

      for (uint iw = 0; iw < input_width; iw++) {
        uint ow0 = start_index(iw, input_width, output_width);
        uint ow1 = end_index(iw, input_width, output_width);
        uint kw = ow1 - ow0;

        float delta = input_ptr[ih * input_width + iw] / __uint2float_rn(kh * kw);
        for (uint oh = oh0; oh < oh1; oh++) {
          for (uint ow = ow0; ow < ow1; ow++) {
            output_ptr[oh * output_width + ow] += delta;
          }
        }
      }
    }
  }
}