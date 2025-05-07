package com.omega.engine.nn.layer.active.gpu;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class TanhKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction function_back;
    private CUfunction function_back_temp;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public TanhKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 8;
        float[] x1 = new float[]{1, 2, 3, 4, -5, 6, -7, -8, 9, 10, 11, -12, 13, 14, 15, -16};
        float[] bias1 = MatrixUtils.one(x1.length);
        Tensor input = new Tensor(N, C, H, W, x1, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor delta = new Tensor(N, C, H, W, bias1, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        TanhKernel k = new TanhKernel(cudaManager);
        k.forward(input, output);
        k.backward(output, delta, diff);
        output.showDM();
        diff.showDM();
        k.backward(output, delta, diff);
        diff.showDM();
        CUDAMemoryManager.free();
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
                function = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "tanh_forward");
            }
            if (function_back == null) {
                function_back = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "tanh_backward");
            }
            if (function_back_temp == null) {
                function_back_temp = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "tanh_backward_temp");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor input, Tensor output, int index, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(output.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(new int[]{length}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Pointer input, Pointer output, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input), Pointer.to(output), Pointer.to(new int[]{length}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Tensor input, Tensor output) {
        try {
            //			if(forwardKernelParameters == null || this.N != output.number) {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            //			}
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor output, Tensor delta, Tensor diff) {
        try {
            //			if(backwardKernelParameters == null) {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            //			}
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, int step) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)), Pointer.to(delta.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)), Pointer.to(diff.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)), Pointer.to(new int[]{input.getOnceSize()}));
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(input.getOnceSize()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardTemp(Tensor output, Tensor delta, Tensor diff) {
        try {
            //			if(backwardKernelParameters == null) {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            //			}
            cuLaunchKernel(function_back_temp, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor output, Tensor delta, Tensor diff, int index, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(output.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(delta.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(diff.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(new int[]{length}));
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Pointer output, Pointer delta, Pointer diff, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(output), Pointer.to(delta), Pointer.to(diff), Pointer.to(new int[]{length}));
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

