package com.omega.engine.nn.layer.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AVGPoolingKernel extends BaseKernel {
    private int C;
    private int H;
    private int W;
    private CUfunction forward_function;
    private CUfunction backward_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public AVGPoolingKernel(int C, int H, int W, CUDAManager cudaManager) {
        super(cudaManager);
        this.C = C;
        this.H = H;
        this.W = W;
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 3;
        int H = 4;
        int W = 4;
        int oHeight = 1;
        int oWidth = 1;
        float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        float[] d = RandomUtils.order(N * C * oHeight * oWidth, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor output = new Tensor(N, C, oHeight, oWidth, true);
        Tensor delta = new Tensor(N, C, oHeight, oWidth, d, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        AVGPoolingKernel pooling = new AVGPoolingKernel(C, H, W, cudaManager);
        long start = System.nanoTime();
        for (int i = 0; i < 2; i++) {
            pooling.forward(input, output);
        }
        System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
        output.showDM();
        input.showDM();
        pooling.backward(delta, diff);
        delta.showDM();
        diff.showDM();
        //    	System.out.println(JsonUtils.toJson(out));
        //    	System.out.println(JsonUtils.toJson(mask));
        //	    System.out.println(JsonUtils.toJson(out));
        //
        //	    System.out.println(JsonUtils.toJson(x));
        //
        //	    System.out.println(JsonUtils.toJson(xout));
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("AVGPoolingKernel.cu", "pooling_forward");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("AVGPoolingKernel.cu", "pooling_backward");
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

    public void forward(Tensor input, Tensor output) {
        pooling(input, output);
    }

    public void backward(Tensor delta, Tensor diff) {
        poolingDiff(delta, diff);
    }

    public void pooling(Tensor input, Tensor output) {
        try {
            //			long start1 = System.nanoTime();
            if (input.number != this.N) {
                this.N = input.number;
                /**
                 * 设置入参
                 * int n, int w, int h, int c, float *input, float *output

                 */
                forwardKernelParameters = Pointer.to(Pointer.to(new int[]{C * N}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H}), Pointer.to(new int[]{C}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(C * N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void poolingDiff(Tensor delta, Tensor diff) {
        try {
            if (backwardKernelParameters == null) {
                /**
                 * 设置入参
                 * int n, int w, int h, int c, float *in_delta, float *out_delta

                 */
                backwardKernelParameters = Pointer.to(Pointer.to(new int[]{C * N}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H}), Pointer.to(new int[]{C}), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()));
            }
            cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(C * N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
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

