package com.omega.engine.nn.network.utils;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.util.List;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

public class ClipGradNormKernel extends CUDAKernel {

    private static final int THREADS = 256;

    private CUfunction sumSqFunction;
    private CUfunction normFunction;
    private CUfunction coefFunction;
    private CUfunction scaleFunction;

    private Pointer dGradPtrs;
    private Pointer dSizes;
    private Pointer dOffsets;
    private Pointer dSumSq;
    private Pointer dCoef;
    private Tensor norm;

    private int tensorCapacity = 0;
    private long totalCapacity = 0;

    public ClipGradNormKernel(CUDAManager cudaManager) {
        super(cudaManager);
        sumSqFunction = cudaManager.getLocalFunctionByModule("ClipGradNormKernel.cu", "grad_global_sum_sq_kernel");
        normFunction = cudaManager.getLocalFunctionByModule("ClipGradNormKernel.cu", "grad_norm_from_sum_sq_kernel");
        coefFunction = cudaManager.getLocalFunctionByModule("ClipGradNormKernel.cu", "grad_clip_coef_kernel");
        scaleFunction = cudaManager.getLocalFunctionByModule("ClipGradNormKernel.cu", "grad_scale_kernel");

        dSumSq = new Pointer();
        dCoef = new Pointer();
        JCuda.cudaMalloc(dSumSq, Sizeof.FLOAT);
        JCuda.cudaMalloc(dCoef, Sizeof.FLOAT);
        norm = new Tensor(1, 1, 1, 1, true);
    }

    public Tensor norm(List<Tensor> grads) {
        return norm(grads, norm);
    }

    public Tensor norm(List<Tensor> grads, Tensor output) {
        output = Tensor.createGPUTensor(output, 1, 1, 1, 1, true);
        if (grads == null || grads.isEmpty()) {
            output.clearGPU();
            return output;
        }

        long totalSize = prepare(grads);
        if (totalSize <= 0L) {
            output.clearGPU();
            return output;
        }

        launchSumSq(grads.size(), totalSize);
        launchNorm(output);
        return output;
    }

    public void clip(List<Tensor> grads, float maxNorm) {
        clip(grads, maxNorm, 1e-6f);
    }

    public void clip(List<Tensor> grads, float maxNorm, float eps) {
        if (grads == null || grads.isEmpty() || maxNorm <= 0.0f) {
            return;
        }

        long totalSize = prepare(grads);
        if (totalSize <= 0L) {
            return;
        }

        int tensorCount = grads.size();
        int blocks = launchSumSq(tensorCount, totalSize);
        launchClipCoef(maxNorm, eps);
        launchScale(tensorCount, totalSize, blocks);
    }

    private long prepare(List<Tensor> grads) {
        int tensorCount = grads.size();
        int[] sizes = new int[tensorCount];
        long[] offsets = new long[tensorCount];
        Pointer[] ptrs = new Pointer[tensorCount];

        long totalSize = 0L;
        for (int i = 0; i < tensorCount; i++) {
            Tensor g = grads.get(i);
            ptrs[i] = g.getGpuData();
            sizes[i] = g.dataLength;
            offsets[i] = totalSize;
            totalSize += g.dataLength;
        }

        ensureCapacity(tensorCount, totalSize);

        JCuda.cudaMemcpy(dGradPtrs, Pointer.to(ptrs), (long) tensorCount * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(dSizes, Pointer.to(sizes), (long) tensorCount * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(dOffsets, Pointer.to(offsets), (long) tensorCount * Sizeof.LONG, cudaMemcpyKind.cudaMemcpyHostToDevice);
        return totalSize;
    }

    private int launchSumSq(int tensorCount, long totalSize) {
        JCuda.cudaMemset(dSumSq, 0, Sizeof.FLOAT);

        int blocks = (int) Math.min(65535L, (totalSize + THREADS - 1L) / THREADS);

        Pointer sumParams = Pointer.to(
                Pointer.to(dGradPtrs),
                Pointer.to(dSizes),
                Pointer.to(dOffsets),
                Pointer.to(new int[]{tensorCount}),
                Pointer.to(new long[]{totalSize}),
                Pointer.to(dSumSq)
        );

        checkCUDA(cuLaunchKernel(
                sumSqFunction,
                blocks, 1, 1,
                THREADS, 1, 1,
                0, null,
                sumParams, null
        ));
        return blocks;
    }

    private void launchNorm(Tensor output) {
        Pointer normParams = Pointer.to(
                Pointer.to(dSumSq),
                Pointer.to(output.getGpuData())
        );

        checkCUDA(cuLaunchKernel(
                normFunction,
                1, 1, 1,
                1, 1, 1,
                0, null,
                normParams, null
        ));
    }

    private void launchClipCoef(float maxNorm, float eps) {
        Pointer coefParams = Pointer.to(
                Pointer.to(dSumSq),
                Pointer.to(dCoef),
                Pointer.to(new float[]{maxNorm}),
                Pointer.to(new float[]{eps})
        );

        checkCUDA(cuLaunchKernel(
                coefFunction,
                1, 1, 1,
                1, 1, 1,
                0, null,
                coefParams, null
        ));
    }

    private void launchScale(int tensorCount, long totalSize, int blocks) {
        Pointer scaleParams = Pointer.to(
                Pointer.to(dGradPtrs),
                Pointer.to(dSizes),
                Pointer.to(dOffsets),
                Pointer.to(new int[]{tensorCount}),
                Pointer.to(new long[]{totalSize}),
                Pointer.to(dCoef)
        );

        checkCUDA(cuLaunchKernel(
                scaleFunction,
                blocks, 1, 1,
                THREADS, 1, 1,
                0, null,
                scaleParams, null
        ));
    }

    private void ensureCapacity(int tensorCount, long totalSize) {
        if (tensorCount > tensorCapacity) {
            if (dGradPtrs != null) JCuda.cudaFree(dGradPtrs);
            if (dSizes != null) JCuda.cudaFree(dSizes);
            if (dOffsets != null) JCuda.cudaFree(dOffsets);

            dGradPtrs = new Pointer();
            dSizes = new Pointer();
            dOffsets = new Pointer();

            JCuda.cudaMalloc(dGradPtrs, (long) tensorCount * Sizeof.POINTER);
            JCuda.cudaMalloc(dSizes, (long) tensorCount * Sizeof.INT);
            JCuda.cudaMalloc(dOffsets, (long) tensorCount * Sizeof.LONG);

            tensorCapacity = tensorCount;
        }

        totalCapacity = Math.max(totalCapacity, totalSize);
    }

    public void free() {
        if (dGradPtrs != null) JCuda.cudaFree(dGradPtrs);
        if (dSizes != null) JCuda.cudaFree(dSizes);
        if (dOffsets != null) JCuda.cudaFree(dOffsets);
        if (dSumSq != null) JCuda.cudaFree(dSumSq);
        if (dCoef != null) JCuda.cudaFree(dCoef);

        dGradPtrs = null;
        dSizes = null;
        dOffsets = null;
        dSumSq = null;
        dCoef = null;
        tensorCapacity = 0;
        totalCapacity = 0;
    }
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}
