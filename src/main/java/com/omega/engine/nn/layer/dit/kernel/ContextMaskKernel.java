package com.omega.engine.nn.layer.dit.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

/**
 * Implements context = where(attnMask == 1, context, maskToken).
 * The mask is flattened as [B * T], while context is [B * T, hiddenSize].
 */
public class ContextMaskKernel extends CUDAKernel {

    private static final int THREADS = 256;
    private static final int MAX_REDUCTION_PARTITIONS = 32;

    private final CUfunction fuseMaskFunction;
    private final CUfunction forwardFunction;
    private final CUfunction backwardFunction;
    private final CUfunction maskTokenBackwardFunction;

    public ContextMaskKernel(CUDAManager cudaManager) {
        super(cudaManager);
        fuseMaskFunction = cudaManager.getLocalFunctionByModule(
                "ContextMaskKernel.cu", "fuse_context_attn_mask");
        forwardFunction = cudaManager.getLocalFunctionByModule(
                "ContextMaskKernel.cu", "context_mask_forward");
        backwardFunction = cudaManager.getLocalFunctionByModule(
                "ContextMaskKernel.cu", "context_mask_backward");
        maskTokenBackwardFunction = cudaManager.getLocalFunctionByModule(
                "ContextMaskKernel.cu", "context_mask_token_backward");
    }

    /**
     * A dropped sample becomes padding at every token; other samples retain
     * their original T5 mask. Mask values are 1 for valid and 0 for padding.
     */
    public void fuseAttnMask(Tensor attnMask, Tensor dropMask, Tensor fusedMask,
                             int tokens, float dropProbability) {
        requireGpu("attnMask", attnMask);
        requireGpu("dropMask", dropMask);
        requireGpu("fusedMask", fusedMask);
        if (tokens <= 0 || attnMask.dataLength % tokens != 0) {
            throw new IllegalArgumentException("attnMask size must be divisible by tokens.");
        }
        int batch = attnMask.dataLength / tokens;
        if (dropMask.dataLength != batch || fusedMask.dataLength != attnMask.dataLength) {
            throw new IllegalArgumentException(
                    "Expected dropMask [B] and fusedMask [B,T] for B=" + batch
                            + ", T=" + tokens + ".");
        }
        if (!Float.isFinite(dropProbability)
                || dropProbability < 0.0f || dropProbability > 1.0f) {
            throw new IllegalArgumentException("dropProbability must be in [0, 1].");
        }

        Pointer parameters = Pointer.to(
                Pointer.to(attnMask.getGpuData()),
                Pointer.to(dropMask.getGpuData()),
                Pointer.to(fusedMask.getGpuData()),
                Pointer.to(new int[]{attnMask.dataLength}),
                Pointer.to(new int[]{tokens}),
                Pointer.to(new float[]{dropProbability})
        );
        CUDAManager.checkCUDA(cuLaunchKernel(
                fuseMaskFunction,
                blocks(attnMask.dataLength), 1, 1,
                THREADS, 1, 1,
                0, null,
                parameters, null
        ));
    }

    /**
     * output may be the same tensor as context.
     */
    public void forward(Tensor context, Tensor attnMask, Tensor maskToken, Tensor output) {
        int hiddenSize = validateForward(context, attnMask, maskToken, output);
        Pointer parameters = Pointer.to(
                Pointer.to(context.getGpuData()),
                Pointer.to(attnMask.getGpuData()),
                Pointer.to(maskToken.getGpuData()),
                Pointer.to(output.getGpuData()),
                Pointer.to(new long[]{context.dataLength}),
                Pointer.to(new int[]{hiddenSize})
        );

        CUDAManager.checkCUDA(cuLaunchKernel(
                forwardFunction,
                blocks(context.dataLength), 1, 1,
                THREADS, 1, 1,
                0, null,
                parameters, null
        ));
    }

    /**
     * Computes both gradients of torch.where. diffMaskToken is overwritten.
     * diffContext may alias delta.
     */
    public void backward(Tensor delta, Tensor attnMask, Tensor diffContext,
                         Tensor diffMaskToken) {
        int hiddenSize = validateBackward(delta, attnMask, diffContext, diffMaskToken);
        int rows = attnMask.dataLength;
        int partitions = Math.min(rows, MAX_REDUCTION_PARTITIONS);

        diffMaskToken.clearGPU();
        Pointer parameters = Pointer.to(
                Pointer.to(delta.getGpuData()),
                Pointer.to(attnMask.getGpuData()),
                Pointer.to(diffContext.getGpuData()),
                Pointer.to(diffMaskToken.getGpuData()),
                Pointer.to(new int[]{rows}),
                Pointer.to(new int[]{hiddenSize})
        );

        CUDAManager.checkCUDA(cuLaunchKernel(
                backwardFunction,
                blocks(hiddenSize), partitions, 1,
                THREADS, 1, 1,
                0, null,
                parameters, null
        ));
    }

    /**
     * Computes only d(maskToken) = sum(delta at padding positions).
     * diffMaskToken is overwritten.
     */
    public void backwardMaskToken(Tensor delta, Tensor attnMask, Tensor diffMaskToken) {
        int hiddenSize = validateMaskTokenBackward(delta, attnMask, diffMaskToken);
        int rows = attnMask.dataLength;
        int partitions = Math.min(rows, MAX_REDUCTION_PARTITIONS);

        diffMaskToken.clearGPU();
        Pointer parameters = Pointer.to(
                Pointer.to(delta.getGpuData()),
                Pointer.to(attnMask.getGpuData()),
                Pointer.to(diffMaskToken.getGpuData()),
                Pointer.to(new int[]{rows}),
                Pointer.to(new int[]{hiddenSize})
        );

        CUDAManager.checkCUDA(cuLaunchKernel(
                maskTokenBackwardFunction,
                blocks(hiddenSize), partitions, 1,
                THREADS, 1, 1,
                0, null,
                parameters, null
        ));
    }

    private int validateForward(Tensor context, Tensor attnMask, Tensor maskToken,
                                Tensor output) {
        requireGpu("context", context);
        requireGpu("attnMask", attnMask);
        requireGpu("maskToken", maskToken);
        requireGpu("output", output);
        if (context.dataLength != output.dataLength) {
            throw new IllegalArgumentException("context and output sizes must match.");
        }
        int hiddenSize = maskToken.dataLength;
        validateRows(context.dataLength, attnMask.dataLength, hiddenSize);
        return hiddenSize;
    }

    private int validateBackward(Tensor delta, Tensor attnMask, Tensor diffContext,
                                 Tensor diffMaskToken) {
        requireGpu("diffContext", diffContext);
        if (delta.dataLength != diffContext.dataLength) {
            throw new IllegalArgumentException("delta and diffContext sizes must match.");
        }
        return validateMaskTokenBackward(delta, attnMask, diffMaskToken);
    }

    private int validateMaskTokenBackward(Tensor delta, Tensor attnMask,
                                          Tensor diffMaskToken) {
        requireGpu("delta", delta);
        requireGpu("attnMask", attnMask);
        requireGpu("diffMaskToken", diffMaskToken);
        int hiddenSize = diffMaskToken.dataLength;
        validateRows(delta.dataLength, attnMask.dataLength, hiddenSize);
        return hiddenSize;
    }

    private void validateRows(int contextElements, int maskElements, int hiddenSize) {
        if (maskElements <= 0 || hiddenSize <= 0
                || (long) maskElements * hiddenSize != contextElements) {
            throw new IllegalArgumentException(
                    "Expected context elements == attnMask elements * hiddenSize, got "
                            + contextElements + " != " + maskElements + " * " + hiddenSize);
        }
    }

    private void requireGpu(String name, Tensor tensor) {
        if (tensor == null || !tensor.isHasGPU()) {
            throw new IllegalArgumentException(name + " must be a non-null GPU tensor.");
        }
    }

    private int blocks(int elements) {
        return (elements + THREADS - 1) / THREADS;
    }
}
