package com.omega.engine.nn.layer.active.gpu;

import com.omega.common.config.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class GeluKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction oldFunction;
    private CUfunction oldHalfFunction;
    private CUfunction function_back;
    private CUfunction fast_function;
    private CUfunction fast_function_back;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public GeluKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 2;
        //	    	float[] x1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        float[] x1 = new float[]{-3.45117188f, 1f, -2.45117188f, 1f};
        Tensor input = new Tensor(N, C, H, W, x1, true);
        Tensor output = new Tensor(N, C, H, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        GeluKernel k = new GeluKernel(cudaManager);
        input.showDM();
        k.oldHalfForward(input, output);
        output.showDM();
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
                function = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_forward");
            }
            if (oldFunction == null) {
                oldFunction = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_old_forward");
            }
            if (oldHalfFunction == null) {
                oldHalfFunction = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_old_half_forward");
            }
            if (function_back == null) {
                function_back = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_backward");
            }
            if (fast_function == null) {
                fast_function = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_fwd_cuda");
            }
            if (fast_function_back == null) {
                fast_function_back = getCudaManager().getLocalFunctionByModule("activeFunction.cu", "gelu_bwd_cuda");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor input, Tensor output) {
        try {
            //			if(forwardKernelParameters == null || this.N != output.number) {
            /**
             * 设置入参
             * float *x, float *output, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            //			}
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //			JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void oldForward(Tensor input, Tensor output) {
        try {
            /**
             * 设置入参
             * float *x, float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            cuLaunchKernel(oldFunction, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void oldHalfForward(Tensor input, Tensor output) {
        try {
            /**
             * 设置入参
             * float *x, float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.dataLength}));
            this.N = output.number;
            cuLaunchKernel(oldHalfFunction, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fast_forward(Tensor input, Tensor output) {
        try {
            //			if(forwardKernelParameters == null || this.N != output.number) {
            /**
             * 设置入参
             * float *x, float *output, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            //			}
            cuLaunchKernel(fast_function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //			JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
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

    public void backward(Tensor input, Tensor delta, Tensor diff) {
        try {
            //			if(backwardKernelParameters == null) {
            /**
             * 设置入参
             * float* dinp, const float* inp, const float* dout, int N

             */
            backwardKernelParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{input.dataLength}));
            //			}
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fast_backward(Tensor input, Tensor delta, Tensor diff) {
        try {
            //			if(backwardKernelParameters == null) {
            /**
             * 设置入参
             * float* dinp, const float* inp, const float* dout, int N

             */
            backwardKernelParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{input.dataLength}));
            //			}
            cuLaunchKernel(fast_function_back, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, int index, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(delta.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(diff.getGpuData().withByteOffset(index * Sizeof.FLOAT)), Pointer.to(new int[]{length}));
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

    public void backward(Pointer input, Pointer delta, Pointer diff, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            backwardKernelParameters = Pointer.to(Pointer.to(input), Pointer.to(delta), Pointer.to(diff), Pointer.to(new int[]{length}));
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

