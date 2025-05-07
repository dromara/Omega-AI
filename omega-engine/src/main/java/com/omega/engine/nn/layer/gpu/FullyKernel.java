package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class FullyKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction function_bias;
    private CUfunction back_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private Pointer kernelBackParameters;

    public FullyKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 8;
        float[] x1 = RandomUtils.order(N * C * H * W, 0.0000001f, 0.0000001f);
        float[] x2 = RandomUtils.order(N * C * H * W, 0.0000001f, 0.0000001f);
        float[] d = RandomUtils.order(N * C * H * W, 0.0001f, 0.0001f);
        float[] bias1 = RandomUtils.order(H * W, 0.000001f, 0.00001f);
        float[] bias2 = RandomUtils.order(H * W, 0.000001f, 0.00001f);
        Tensor output = new Tensor(N, C, H, W, x1, true);
        Tensor output2 = new Tensor(N, C, H, W, x2, true);
        Tensor bias = new Tensor(1, 1, 1, W, bias1, true);
        //	    	Tensor bias3 = new Tensor(1, 1, 1, W, bias2, true);
        Tensor diffB = new Tensor(N, C, H, W, d, true);
        Tensor diffB2 = new Tensor(N, C, H, W, d, true);
        Tensor delta = new Tensor(N, C, H, W, d, true);
        CUDAManager cudaManager = new CUDAManager(0);
        FullyKernel k = new FullyKernel(cudaManager);
        //	    	output.showDM(new float[N * C * H * W]);
        output.showDM();
        k.addBias(output, bias);
        //	    	k.backwardBias(diffB, delta);
        //	    	output.showDM(new float[N * C * H * W]);
        //	    	output.syncHost();
        //
        //	    	System.out.println(JsonUtils.toJson(output.getData()));
        //
        //	    	diffB.syncHost();
        //
        //	    	System.out.println(JsonUtils.toJson(diffB.getData()));
        output.showDM();
        CUDAMemoryManager.free();
        for (int n = 0; n < N; n++) {
            for (int ow = 0; ow < W; ow++) {
                output2.data[n * W + ow] += bias.data[ow];
            }
        }
        //			diffB.showDM();
        System.out.println(JsonUtils.toJson(output2.data));
        k.backwardBias(diffB, delta);
        for (int ow = 0; ow < W; ow++) {
            diffB2.data[ow] = 0.0f;
            for (int n = 0; n < N; n++) {
                diffB2.data[ow] += delta.data[n * W + ow];
            }
        }
        diffB.showDM();
        System.out.println(JsonUtils.toJson(diffB2.data));
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
                function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "add_bias");
            }
            if (function_bias == null) {
                function_bias = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "add_full_bias");
            }
            if (back_function == null) {
                back_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "backward_bias_conn_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
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
            checkCUDA(cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void addBias(Tensor output,Tensor bias,int batch,int step) {
    //
    //		try {
    //
    //	        /**
    //	         * 设置入参
    //	         * float* output, float* biases, int N, int w
    //	         */
    //	        kernelParameters = Pointer.to(
    //	        		Pointer.to(output.getGpuData().withByteOffset(step * batch * output.getOnceSize() * Sizeof.FLOAT)),
    //	                Pointer.to(bias.getGpuData()),
    //	                Pointer.to(new int[]{batch * output.getOnceSize()}),
    //	                Pointer.to(new int[]{output.getWidth()})
    //	            );
    //
    //			cuLaunchKernel(function_bias,
    //		            this.CAFFE_GET_BLOCKS(batch * output.getOnceSize()),  1, 1,      // Grid dimension
    //		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //		            0, null,               // Shared memory size and stream
    //		            kernelParameters, null // Kernel- and extra parameters
    //		        );
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void addBias(Tensor output, Tensor bias, int batch, int step) {
        try {
            /**
             * 设置入参
             * float* output, float* biases, int batch, int n, int size

             */
            kernelParameters = Pointer.to(Pointer.to(output.getGpuData().withByteOffset(step * batch * output.getOnceSize() * Sizeof.FLOAT)), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{batch}), Pointer.to(new int[]{output.getWidth()}), Pointer.to(new int[]{1}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(batch * output.getOnceSize()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardBias(Tensor diffB, Tensor delta) {
        try {
            diffB.clearGPU();
            if (kernelBackParameters == null) {
                /**
                 * 设置入参
                 * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

                 */
                kernelBackParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getNumber()}), Pointer.to(new int[]{delta.getWidth()}));
            }
            cuLaunchKernel(back_function, this.CAFFE_GET_BLOCKS(delta.getWidth()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelBackParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardBias(Tensor diffB, Tensor delta, int batch, int step) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            kernelBackParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData().withByteOffset(step * batch * delta.getOnceSize() * Sizeof.FLOAT)), Pointer.to(new int[]{batch}), Pointer.to(new int[]{delta.getWidth()}));
            cuLaunchKernel(back_function, this.CAFFE_GET_BLOCKS(batch * delta.getWidth()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelBackParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

