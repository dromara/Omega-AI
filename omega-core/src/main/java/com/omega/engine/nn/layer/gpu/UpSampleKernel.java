package com.omega.engine.nn.layer.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class UpSampleKernel extends BaseKernel {
    private int stride;
    private float scale;
    private boolean reverse = false;
    private CUfunction forward_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public UpSampleKernel(int stride, float scale, CUDAManager cudaManager) {
        super(cudaManager);
        this.stride = stride;
        this.scale = scale;
        if (this.stride < 0) {
            this.stride = -stride;
            reverse = true;
        }
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 3;
        int H = 4;
        int W = 4;
        int stride = 2;
        float scale = 1.0f;
        int oHeight = H * stride;
        int oWidth = W * stride;
        if (stride < 0) {
            stride = -stride;
            oHeight = H / stride;
            oWidth = W / stride;
        }
        float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        float[] d = RandomUtils.order(N * C * oHeight * oWidth, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor output = new Tensor(N, C, oHeight, oWidth, true);
        float[] output_cpu = new float[output.getDataLength()];
        Tensor delta = new Tensor(N, C, oHeight, oWidth, d, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        float[] diff_cpu = new float[diff.getDataLength()];
        CUDAManager cudaManager = new CUDAManager(0);
        UpSampleKernel pooling = new UpSampleKernel(stride, scale, cudaManager);
        long start = System.nanoTime();
        //    	for(int i = 0;i<2;i++) {
        pooling.forward(input, output);
        //    	}
        System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
        input.showDM();
        output.showDM();
        upsample_cpu(input.getData(), W, H, C, N, stride, 1, scale, output_cpu);
        System.out.println(JsonUtils.toJson(output_cpu));
        pooling.backward(delta, diff);
        delta.showDM();
        diff.showDM();
        upsample_cpu(diff_cpu, W, H, C, N, stride, 0, scale, delta.getData());
        System.out.println(JsonUtils.toJson(diff_cpu));
        //    	System.out.println(JsonUtils.toJson(out));
        //    	System.out.println(JsonUtils.toJson(mask));
        //	    System.out.println(JsonUtils.toJson(out));
        //
        //	    System.out.println(JsonUtils.toJson(x));
        //
        //	    System.out.println(JsonUtils.toJson(xout));
    }

    public static void upsample_cpu(float[] in, int w, int h, int c, int batch, int stride, int forward, float scale, float[] out) {
        int i, j, k, b;
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < c; ++k) {
                for (j = 0; j < h * stride; ++j) {
                    for (i = 0; i < w * stride; ++i) {
                        int in_index = b * w * h * c + k * w * h + (j / stride) * w + i / stride;
                        int out_index = b * w * h * c * stride * stride + k * w * h * stride * stride + j * w * stride + i;
                        if (forward == 1)
                            out[out_index] = scale * in[in_index];
                        else
                            in[in_index] += scale * out[out_index];
                    }
                }
            }
        }
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel.cu", "upsample_kernel");
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
        if (reverse) {
            upsample(output, input, 0);
        } else {
            upsample(input, output, 1);
        }
    }

    public void backward(Tensor delta, Tensor diff) {
        if (reverse) {
            upsampleDelta(delta, diff, 1);
        } else {
            upsampleDelta(diff, delta, 0);
        }
    }

    public void upsample(Tensor input, Tensor output, int forward) {
        try {
            //			long start1 = System.nanoTime();
            int size = input.getShape()[1] * input.getShape()[0] * input.getShape()[3] * input.getShape()[2] * stride * stride;
            if (input.getShape()[0] != this.N) {
                this.N = input.getShape()[0];
                /**
                 * 设置入参
                 * int N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out

                 */
                forwardKernelParameters = Pointer.to(Pointer.to(new int[]{size}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{input.getShape()[3]}), Pointer.to(new int[]{input.getShape()[2]}), Pointer.to(new int[]{input.getShape()[1]}), Pointer.to(new int[]{input.getShape()[0]}), Pointer.to(new int[]{stride}), Pointer.to(new int[]{forward}), Pointer.to(new float[]{scale}), Pointer.to(output.getGpuData()));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(size), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void upsampleDelta(Tensor delta, Tensor diff, int forward) {
        try {
            int size = delta.getShape()[1] * delta.getShape()[0] * delta.getShape()[3] * delta.getShape()[2] * stride * stride;
            if (backwardKernelParameters == null) {
                /**
                 * 设置入参
                 * int N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out

                 */
                backwardKernelParameters = Pointer.to(Pointer.to(new int[]{size}), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getShape()[3]}), Pointer.to(new int[]{delta.getShape()[2]}), Pointer.to(new int[]{delta.getShape()[1]}), Pointer.to(new int[]{delta.getShape()[0]}), Pointer.to(new int[]{stride}), Pointer.to(new int[]{forward}), Pointer.to(new float[]{scale}), Pointer.to(diff.getGpuData()));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(size), 1, 1,      // Grid dimension
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

