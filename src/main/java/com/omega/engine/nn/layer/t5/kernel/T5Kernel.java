package com.omega.engine.nn.layer.t5.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class T5Kernel extends CUDAKernel {
	
    private CUfunction function;

    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public T5Kernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("T5Kernel.cu", "compute_bias");
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
    
    public static void main(String args[]) {

    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void compute_bias(Tensor x, Tensor mask, Tensor output, int time, int w) {
        try {
      
            /**
             * 设置入参
             * const size_t size, const float *x, float *mask, float *out, const int ht, const int len
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{output.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{time}), Pointer.to(new int[]{w}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

}

