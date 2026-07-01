// ClipGradNormKernel.cu
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK 256

__device__ __forceinline__ float sanitize_grad(float v) {
    return isfinite(v) ? v : 0.0f;
}

extern "C"
__global__ void grad_global_sum_sq_kernel(
    float** grads,
    const int* sizes,
    const long long* offsets,
    int tensorCount,
    long long totalSize,
    float* sumSq
) {
    __shared__ float smem[BLOCK];

    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float local = 0.0f;

    for (long long idx = tid; idx < totalSize; idx += stride) {
        int lo = 0;
        int hi = tensorCount - 1;

        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            long long begin = offsets[mid];
            long long end = begin + sizes[mid];

            if (idx < begin) {
                hi = mid - 1;
            } else if (idx >= end) {
                lo = mid + 1;
            } else {
                float v = sanitize_grad(grads[mid][idx - begin]);
                local += v * v;
                break;
            }
        }
    }

    smem[threadIdx.x] = local;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sumSq, smem[0]);
    }
}

extern "C"
__global__ void grad_clip_coef_kernel(
    const float* sumSq,
    float* coef,
    float maxNorm,
    float eps
) {
    float norm = sqrtf(fmaxf(sumSq[0], 0.0f));
    float c = maxNorm / (norm + eps);
    coef[0] = fminf(c, 1.0f);
}

extern "C"
__global__ void grad_scale_kernel(
    float** grads,
    const int* sizes,
    const long long* offsets,
    int tensorCount,
    long long totalSize,
    const float* coef
) {
    float c = coef[0];

    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long idx = tid; idx < totalSize; idx += stride) {
        int lo = 0;
        int hi = tensorCount - 1;

        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            long long begin = offsets[mid];
            long long end = begin + sizes[mid];

            if (idx < begin) {
                hi = mid - 1;
            } else if (idx >= end) {
                lo = mid + 1;
            } else {
                grads[mid][idx - begin] *= c;
                break;
            }
        }
    }
}