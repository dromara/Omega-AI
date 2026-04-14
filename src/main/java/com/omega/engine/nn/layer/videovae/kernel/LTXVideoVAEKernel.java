package com.omega.engine.nn.layer.videovae.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class LTXVideoVAEKernel extends CUDAKernel {
	
    private CUfunction function;
    
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public LTXVideoVAEKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
//        int N = 1;
//        int C = 1;
//        int D = 2;
//        int H = 3;
//        int W = 3;
//        float[] x = MatrixUtils.order(N * C * D * H * W, 1, 1);
//        Tensor input = new Tensor(N, C * D, H, W, x, true);
//        
//        CUDAManager cudaManager = new CUDAManager(0);
//
//        LTXVideoVAEKernel pad = new LTXVideoVAEKernel(cudaManager);
 
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("ltx_video_vae.cu", "encoder_repeat");
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

    /**
     * padding shape [wLeft,wRight,hTop,hBottom]
     */
    public void encoder_repeat(Tensor x, Tensor y, int C, int F, int H, int W, int xIndex, int count) {
        try {
            /**
             * 设置入参
             *const size_t size, const float *x, float *out, const int C, const int F, const int H, const int W, int xIndex, int count
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{y.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{C}), Pointer.to(new int[]{F}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{xIndex}), Pointer.to(new int[]{count}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(y.dataLength), 1, 1,      // Grid dimension
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

