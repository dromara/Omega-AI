#include <cuda_runtime.h>

extern "C"
__global__ void fuse_context_attn_mask(
    const float* __restrict__ attn_mask,
    const float* __restrict__ drop_mask,
    float* __restrict__ fused_mask,
    int rows,
    int tokens,
    float drop_probability
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += blockDim.x * gridDim.x) {
        int batch = row / tokens;
        fused_mask[row] = drop_mask[batch] < drop_probability
            ? 0.0f
            : attn_mask[row];
    }
}

extern "C"
__global__ void context_mask_forward(
    const float* __restrict__ context,
    const float* __restrict__ attn_mask,
    const float* __restrict__ mask_token,
    float* __restrict__ output,
    long long element_count,
    int hidden_size
) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < element_count;
         index += (long long)blockDim.x * gridDim.x) {
        int row = (int)(index / hidden_size);
        int hidden = (int)(index - (long long)row * hidden_size);
        output[index] = attn_mask[row] == 1.0f
            ? context[index]
            : mask_token[hidden];
    }
}

extern "C"
__global__ void context_mask_backward(
    const float* __restrict__ delta,
    const float* __restrict__ attn_mask,
    float* __restrict__ diff_context,
    float* __restrict__ diff_mask_token,
    int rows,
    int hidden_size
) {
    int hidden = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidden >= hidden_size) {
        return;
    }

    float mask_token_sum = 0.0f;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        long long index = (long long)row * hidden_size + hidden;
        float grad = delta[index];
        if (attn_mask[row] == 1.0f) {
            diff_context[index] = grad;
        } else {
            diff_context[index] = 0.0f;
            mask_token_sum += grad;
        }
    }

    if (mask_token_sum != 0.0f) {
        atomicAdd(diff_mask_token + hidden, mask_token_sum);
    }
}

extern "C"
__global__ void context_mask_token_backward(
    const float* __restrict__ delta,
    const float* __restrict__ attn_mask,
    float* __restrict__ diff_mask_token,
    int rows,
    int hidden_size
) {
    int hidden = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidden >= hidden_size) {
        return;
    }

    float mask_token_sum = 0.0f;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        if (attn_mask[row] != 1.0f) {
            long long index = (long long)row * hidden_size + hidden;
            mask_token_sum += delta[index];
        }
    }

    if (mask_token_sum != 0.0f) {
        atomicAdd(diff_mask_token + hidden, mask_token_sum);
    }
}
