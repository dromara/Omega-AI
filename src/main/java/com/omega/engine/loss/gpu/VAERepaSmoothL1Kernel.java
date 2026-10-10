package com.omega.engine.loss.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

/** Standard mean Smooth L1 used by the isolated VAE-REPA experiment. */
public class VAERepaSmoothL1Kernel extends BaseKernel {

    private static final int BLOCK_SIZE = 1024;

    private CUfunction forwardFunction;
    private CUfunction backwardFunction;

    public VAERepaSmoothL1Kernel(CUDAManager cudaManager) {
        super(cudaManager);
        forwardFunction = cudaManager.getLocalFunctionByModule("VAERepaSmoothL1Kernel.cu", "vae_repa_smooth_l1");
        backwardFunction = cudaManager.getLocalFunctionByModule("VAERepaSmoothL1Kernel.cu", "vae_repa_smooth_l1_back");
    }

    private int blocks(int n) {
        return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    public void forward(Tensor prediction, Tensor target, Tensor loss, float beta) {
        validate(prediction, target, loss, beta);
        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{prediction.dataLength}),
                Pointer.to(prediction.getGpuData()),
                Pointer.to(target.getGpuData()),
                Pointer.to(loss.getGpuData()),
                Pointer.to(new float[]{beta}));
        cuLaunchKernel(forwardFunction, blocks(prediction.dataLength), 1, 1,
                BLOCK_SIZE, 1, 1, 0, null, parameters, null);
    }

    public void backward(Tensor prediction, Tensor target, Tensor diff, float beta) {
        validate(prediction, target, diff, beta);
        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{prediction.dataLength}),
                Pointer.to(prediction.getGpuData()),
                Pointer.to(target.getGpuData()),
                Pointer.to(diff.getGpuData()),
                Pointer.to(new float[]{beta}));
        cuLaunchKernel(backwardFunction, blocks(prediction.dataLength), 1, 1,
                BLOCK_SIZE, 1, 1, 0, null, parameters, null);
    }

    private void validate(Tensor prediction, Tensor target, Tensor result, float beta) {
        if (prediction.dataLength != target.dataLength || prediction.dataLength != result.dataLength) {
            throw new IllegalArgumentException("VAE-REPA tensors must have identical data lengths.");
        }
        if (beta <= 0.0f) {
            throw new IllegalArgumentException("Smooth L1 beta must be positive.");
        }
    }
}
