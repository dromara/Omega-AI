package com.omega.engine.loss.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class SmoothL1Kernel extends BaseKernel {
    private CUfunction loss_function;
    private CUfunction loss_backward_function;
    private CUfunction loss_org_function;
    private CUfunction loss_org_backward_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer loss_kernelParameters;
    private Pointer backKernelParameters;

    public SmoothL1Kernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (loss_function == null) {
                loss_function = getCudaManager().getLocalFunctionByModule("SmoothL1Kernel.cu", "loss");
            }
            if (loss_backward_function == null) {
                loss_backward_function = getCudaManager().getLocalFunctionByModule("SmoothL1Kernel.cu", "loss_back");
            }
            if (loss_org_function == null) {
            	loss_org_function = getCudaManager().getLocalFunctionByModule("SmoothL1Kernel.cu", "loss_org");
            }
            if (loss_org_backward_function == null) {
            	loss_org_backward_function = getCudaManager().getLocalFunctionByModule("SmoothL1Kernel.cu", "loss_org_back");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor input, Tensor currentLabel, Tensor output, float beta) {
        /**
         * int N,float *input, float *label, float *output
         */
        loss_kernelParameters = Pointer.to(Pointer.to(new int[]{input.dataLength}), Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new float[]{beta}));
        cuLaunchKernel(loss_function,  this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                loss_kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward(Tensor input, Tensor currentLabel, Tensor diff, float beta) {
        /**
         * int N,float *input, float *label, float *diff
         */
        backKernelParameters = Pointer.to(Pointer.to(new int[]{input.dataLength}), Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new float[]{beta}));
        cuLaunchKernel(loss_backward_function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                backKernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }
    
    public void forward(Tensor input, Tensor currentLabel, Tensor output) {
        /**
         * int N,float *input, float *label, float *output
         */
        loss_kernelParameters = Pointer.to(Pointer.to(new int[]{input.dataLength}), Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(output.getGpuData()));
        cuLaunchKernel(loss_org_function,  this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                loss_kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward(Tensor input, Tensor currentLabel, Tensor diff, int tmp) {
        /**
         * int N,float *input, float *label, float *diff
         */
        backKernelParameters = Pointer.to(Pointer.to(new int[]{input.dataLength}), Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{tmp}));
        cuLaunchKernel(loss_org_backward_function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                backKernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

