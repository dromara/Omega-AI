package com.omega.engine.nn.layer.gpu;

import com.omega.common.config.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class BasicBlockKernel extends BaseKernel {
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private CUfunction function;
    private Pointer kernelParameters;

    public BasicBlockKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("BaseMathKernel.cu", "add");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void add(Tensor x, Tensor y, Tensor output) {
        try {
            /**
             * 设置入参
             * float* output, float* biases, int batch, int n, int size

             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

