package com.omega.engine.nn.layer.dit.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class PaddingMaskKernel extends CUDAKernel {
    private CUfunction function;
    private CUfunction function2;
    private CUfunction mask_function;
    private CUfunction mask_diff_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public PaddingMaskKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids");
            }
            if (function2 == null) {
            	function2 = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids_back");
            }
            if (mask_function == null) {
            	mask_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_mask");
            }
            if (mask_diff_function == null) {
            	mask_diff_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "mask_diff");
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
    
    public void set_mask(Tensor mask, Tensor out, int W) {
    	 try {
             /**
              * 设置入参
              * const size_t size, const float *mask, float *out, const int W
              */
             kernelParameters = Pointer.to(Pointer.to(new long[]{out.dataLength}), Pointer.to(mask.getGpuData()), Pointer.to(out.getGpuData()), Pointer.to(new int[]{W}));
             cuLaunchKernel(mask_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,      // Grid dimension
                     CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                     0, null,               // Shared memory size and stream
                     kernelParameters, null // Kernel- and extra parameters
             );
         } catch (Exception e) {
             // TODO: handle exception
             e.printStackTrace();
         }
    }
    
    public void forward(Tensor x, Tensor mask, Tensor idskeep, Tensor output, int FT, int T, int W) {
        try {
        	/**
        	 * set mask value
        	 */
        	set_mask(mask, output, W);
            /**
             * 设置入参
             * const size_t size, const float *x, const float *idskeep, float *out, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void backward(Tensor delta, Tensor idskeep, Tensor dx, Tensor dw, int FT, int T, int W) {
    	set_ids_back(dx, delta, idskeep, FT, T, W);
    	mask_diff(delta, dw, delta.number, W);
    }
    
    public void set_ids_back(Tensor dx, Tensor delta, Tensor idskeep, int FT, int T, int W) {
        try {
            /**
             * 设置入参
             * const size_t size, float *dx, const float *idskeep, const float *dout, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(idskeep.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(function2, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mask_diff(Tensor delta, Tensor dw, int rows, int cols) {
        try {
        	int gridSize = cols;
            int BLOCK_SIZE = 256;
            int block_size = Math.min(BLOCK_SIZE, rows);
        	/**
             * 设置入参
             * const float *dout, float *dw, const int rows, conts int cols
             */
            kernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dw.getGpuData()), Pointer.to(new int[]{rows}), Pointer.to(new int[]{cols}));
            cuLaunchKernel(mask_diff_function, gridSize, 1, 1,      // Grid dimension
            		block_size, 1, 1,      // Block dimension
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

