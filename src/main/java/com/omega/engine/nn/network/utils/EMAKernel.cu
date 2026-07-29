#include <cuda_runtime.h>

#define EMA_BLOCK_SIZE 256

extern "C"
__global__ void ema_update_multi_tensor_kernel(
    float** emaParams,
    float** modelParams,
    const int* sizes,
    const long long* blockOffsets,
    int tensorCount,
    long long totalBlocks,
    float decay
) {
    __shared__ int tensorIndex;
    __shared__ long long localBlock;

    const float modelScale = 1.0f - decay;

    for (long long blockId = blockIdx.x; blockId < totalBlocks; blockId += gridDim.x) {
        if (threadIdx.x == 0) {
            int lo = 0;
            int hi = tensorCount - 1;
            int found = -1;

            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                long long begin = blockOffsets[mid];
                long long blockCount =
                    ((long long)sizes[mid] + EMA_BLOCK_SIZE - 1LL) / EMA_BLOCK_SIZE;
                long long end = begin + blockCount;

                if (blockId < begin) {
                    hi = mid - 1;
                } else if (blockId >= end) {
                    lo = mid + 1;
                } else {
                    found = mid;
                    localBlock = blockId - begin;
                    break;
                }
            }

            tensorIndex = found;
        }
        __syncthreads();

        int tensor = tensorIndex;
        if (tensor >= 0) {
            long long element = localBlock * blockDim.x + threadIdx.x;
            if (element < sizes[tensor]) {
                float emaValue = emaParams[tensor][element];
                float modelValue = modelParams[tensor][element];
                emaParams[tensor][element] = decay * emaValue + modelScale * modelValue;
            }
        }
        __syncthreads();
    }
}
