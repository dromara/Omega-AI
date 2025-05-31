package com.omega.engine.nn.layer.gpu;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * BiasKernel
 *
 * @author Administrator
 */
public class BiasKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction fast_function;
    private CUfunction back_function;
    private CUfunction back_conv_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private Pointer biasConvKernelParameters;
    private Pointer backKernelParameters;
    private Pointer backConvKernelParameters;

    public BiasKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void init() {
        initFunction();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "add_bias");
            }
            if (fast_function == null) {
                fast_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "add_bias_kernel");
            }
            if (back_function == null) {
                back_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "backward_bias_conn_kernel");
            }
            if (back_conv_function == null) {
                back_conv_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "backward_bias_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void addBias(Tensor output, Tensor bias) {
        try {
            if (kernelParameters == null || output.number != this.N) {
                /**
                 * 设置入参
                 * float* output, float* biases, int batch, int n, int size

                 */
                kernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{output.getNumber()}), Pointer.to(new int[]{output.getWidth()}), Pointer.to(new int[]{1}));
                this.N = output.number;
            }
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

    public void addConvBias(Tensor output, Tensor bias) {
        try {
            //			if(biasConvKernelParameters == null || output.number != this.N){
            /**
             * 设置入参
             * float* output, float* biases, int batch, int n, int size

             */
            biasConvKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{output.getNumber()}), Pointer.to(new int[]{output.channel}), Pointer.to(new int[]{output.height * output.width}));
            this.N = output.number;
            //			}
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    biasConvKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void addConvBiasFast(Tensor output, Tensor bias) {
        try {
            if (biasConvKernelParameters == null || output.number != this.N) {
                /**
                 * 设置入参
                 * float *output, float *biases, int n, int size

                 */
                biasConvKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{output.channel}), Pointer.to(new int[]{output.height * output.width}));
                this.N = output.number;
            }
            cuLaunchKernel(fast_function, this.CAFFE_GET_BLOCKS(output.height * output.width), output.channel, output.getNumber(),      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    biasConvKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void addConvBiasFast(Tensor output, Tensor bias,int channel,int depth) {
        try {
            this.N = output.number;
        	 /**
             * 设置入参
             * float *output, float *biases, int n, int size
             */
            biasConvKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{channel}), Pointer.to(new int[]{depth * output.height * output.width}));

            cuLaunchKernel(fast_function, this.CAFFE_GET_BLOCKS(depth * output.height * output.width), channel, output.getNumber(),      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    biasConvKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardBias(Tensor diffB, Tensor delta) {
        try {
            diffB.clearGPU();
            if (backKernelParameters == null || delta.number != this.N) {
                /**
                 * 设置入参
                 * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

                 */
                backKernelParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getNumber()}), Pointer.to(new int[]{delta.getWidth()}));
                this.N = delta.number;
            }
            cuLaunchKernel(back_function, this.CAFFE_GET_BLOCKS(delta.getWidth()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardConvBias(Tensor diffB, Tensor delta) {
        try {
            diffB.clearGPU();
            /**
             * 设置入参
             * float *bias_updates, float *delta, int batch, int n, int size

             */
            backConvKernelParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getNumber()}), Pointer.to(new int[]{delta.getChannel()}), Pointer.to(new int[]{delta.height * delta.width}));
            this.N = delta.number;
            cuLaunchKernel(back_conv_function, delta.getChannel(), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backConvKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardConv3DBias(int depth, Tensor diffB, Tensor delta) {
        try {
            diffB.clearGPU();
            /**
             * 设置入参
             * float *bias_updates, float *delta, int batch, int n, int size

             */
            backConvKernelParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getNumber()}), Pointer.to(new int[]{delta.getChannel() / depth}), Pointer.to(new int[]{depth * delta.height * delta.width}));
            this.N = delta.number;
            cuLaunchKernel(back_conv_function, delta.getChannel() / depth, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backConvKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
}

