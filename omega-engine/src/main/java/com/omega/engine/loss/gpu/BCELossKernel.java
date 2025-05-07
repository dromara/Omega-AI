package com.omega.engine.loss.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class BCELossKernel extends BaseKernel {
    private CUfunction loss_function;
    private CUfunction loss_backward_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer loss_kernelParameters;
    private Pointer backKernelParameters;
    private float eta = 0.0000000001f;

    public BCELossKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (loss_function == null) {
                loss_function = getCudaManager().getLocalFunctionByModule("BECLossKernel.cu", "loss");
            }
            if (loss_backward_function == null) {
                loss_backward_function = getCudaManager().getLocalFunctionByModule("BECLossKernel.cu", "loss_back");
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

    public void forward(Tensor input, Tensor currentLabel, Tensor output) {
        /**
         * float *input, float *label, float *output, int batch, int n, float eta

         */
        loss_kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.number * input.channel}), Pointer.to(new int[]{input.width * input.height}), Pointer.to(new float[]{eta}));
        this.N = output.number;
        cuLaunchKernel(loss_function, input.number * input.channel, 1, 1,    // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                loss_kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward(Tensor input, Tensor currentLabel, Tensor diff) {
        /**
         * float *input, float *currentLabel, float *diff, int batch, int n

         */
        backKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{input.number * input.channel}), Pointer.to(new int[]{input.width * input.height}));
        cuLaunchKernel(loss_backward_function, input.number, 1, 1,      // Grid dimension
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

