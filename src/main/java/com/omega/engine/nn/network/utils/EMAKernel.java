package com.omega.engine.nn.network.utils;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.util.List;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUresult;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class EMAKernel extends CUDAKernel {

    private static final int THREADS = 256;
    private static final int MAX_BLOCKS = 65535;

    private final CUfunction updateFunction;

    private Pointer dEmaPtrs;
    private Pointer dModelPtrs;
    private Pointer dSizes;
    private Pointer dBlockOffsets;

    private Pointer[] emaPtrs;
    private Pointer[] modelPtrs;
    private int[] sizes;
    private long[] blockOffsets;
    private int tensorCapacity;

    public EMAKernel(CUDAManager cudaManager) {
        super(cudaManager);
        updateFunction = cudaManager.getLocalFunctionByModule("EMAKernel.cu", "ema_update_multi_tensor_kernel");
    }

    public void update(List<Tensor> emaParams, List<Tensor> modelParams, float decay) {
        if (emaParams == null || modelParams == null || emaParams.isEmpty()) {
            return;
        }
        if (emaParams.size() != modelParams.size()) {
            throw new IllegalArgumentException("EMA and model parameter counts must match.");
        }
        if (!Float.isFinite(decay) || decay < 0.0f || decay > 1.0f) {
            throw new IllegalArgumentException("EMA decay must be finite and in [0, 1], got: " + decay);
        }

        int tensorCount = emaParams.size();
        ensureCapacity(tensorCount);

        long totalBlocks = prepare(emaParams, modelParams);
        if (totalBlocks == 0L) {
            return;
        }

        uploadMetadata(tensorCount);
        launch(tensorCount, totalBlocks, decay);
    }

    private long prepare(List<Tensor> emaParams, List<Tensor> modelParams) {
        long totalBlocks = 0L;

        for (int i = 0; i < emaParams.size(); i++) {
            Tensor ema = emaParams.get(i);
            Tensor model = modelParams.get(i);

            if (ema == null || model == null) {
                throw new IllegalArgumentException("EMA parameter pair " + i + " contains null.");
            }
            if (!ema.isHasGPU() || !model.isHasGPU()) {
                throw new IllegalArgumentException("EMA parameter pair " + i + " must be on GPU.");
            }
            if (ema.dataLength != model.dataLength) {
                throw new IllegalArgumentException(
                        "EMA parameter pair " + i + " has different sizes: "
                                + ema.dataLength + " vs " + model.dataLength);
            }

            emaPtrs[i] = ema.getGpuData();
            modelPtrs[i] = model.getGpuData();
            sizes[i] = ema.dataLength;
            blockOffsets[i] = totalBlocks;
            totalBlocks += (ema.dataLength + THREADS - 1L) / THREADS;
        }

        return totalBlocks;
    }

    private void uploadMetadata(int tensorCount) {
        CUDAManager.checkCUDA(JCuda.cudaMemcpy(
                dEmaPtrs,
                Pointer.to(emaPtrs),
                (long) tensorCount * Sizeof.POINTER,
                cudaMemcpyKind.cudaMemcpyHostToDevice));
        CUDAManager.checkCUDA(JCuda.cudaMemcpy(
                dModelPtrs,
                Pointer.to(modelPtrs),
                (long) tensorCount * Sizeof.POINTER,
                cudaMemcpyKind.cudaMemcpyHostToDevice));
        CUDAManager.checkCUDA(JCuda.cudaMemcpy(
                dSizes,
                Pointer.to(sizes),
                (long) tensorCount * Sizeof.INT,
                cudaMemcpyKind.cudaMemcpyHostToDevice));
        CUDAManager.checkCUDA(JCuda.cudaMemcpy(
                dBlockOffsets,
                Pointer.to(blockOffsets),
                (long) tensorCount * Sizeof.LONG,
                cudaMemcpyKind.cudaMemcpyHostToDevice));
    }

    private void launch(int tensorCount, long totalBlocks, float decay) {
        int blocks = (int) Math.min(totalBlocks, MAX_BLOCKS);
        Pointer parameters = Pointer.to(
                Pointer.to(dEmaPtrs),
                Pointer.to(dModelPtrs),
                Pointer.to(dSizes),
                Pointer.to(dBlockOffsets),
                Pointer.to(new int[]{tensorCount}),
                Pointer.to(new long[]{totalBlocks}),
                Pointer.to(new float[]{decay})
        );

        int result = cuLaunchKernel(
                updateFunction,
                blocks, 1, 1,
                THREADS, 1, 1,
                0, null,
                parameters, null
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new IllegalStateException("Failed to launch EMA CUDA kernel, error code: " + result);
        }
    }

    private void ensureCapacity(int tensorCount) {
        if (tensorCount <= tensorCapacity) {
            return;
        }

        freeMetadata();

        dEmaPtrs = new Pointer();
        dModelPtrs = new Pointer();
        dSizes = new Pointer();
        dBlockOffsets = new Pointer();

        CUDAManager.checkCUDA(JCuda.cudaMalloc(dEmaPtrs, (long) tensorCount * Sizeof.POINTER));
        CUDAManager.checkCUDA(JCuda.cudaMalloc(dModelPtrs, (long) tensorCount * Sizeof.POINTER));
        CUDAManager.checkCUDA(JCuda.cudaMalloc(dSizes, (long) tensorCount * Sizeof.INT));
        CUDAManager.checkCUDA(JCuda.cudaMalloc(dBlockOffsets, (long) tensorCount * Sizeof.LONG));

        emaPtrs = new Pointer[tensorCount];
        modelPtrs = new Pointer[tensorCount];
        sizes = new int[tensorCount];
        blockOffsets = new long[tensorCount];
        tensorCapacity = tensorCount;
    }

    public void free() {
        freeMetadata();
        emaPtrs = null;
        modelPtrs = null;
        sizes = null;
        blockOffsets = null;
        tensorCapacity = 0;
    }

    private void freeMetadata() {
        if (dEmaPtrs != null) {
            JCuda.cudaFree(dEmaPtrs);
        }
        if (dModelPtrs != null) {
            JCuda.cudaFree(dModelPtrs);
        }
        if (dSizes != null) {
            JCuda.cudaFree(dSizes);
        }
        if (dBlockOffsets != null) {
            JCuda.cudaFree(dBlockOffsets);
        }

        dEmaPtrs = null;
        dModelPtrs = null;
        dSizes = null;
        dBlockOffsets = null;
    }
}
