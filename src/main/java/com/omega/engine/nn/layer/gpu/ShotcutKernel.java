package com.omega.engine.nn.layer.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ShotcutKernel extends BaseKernel {
    private CUfunction function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private float s1 = 1.0f;
    private float s2 = 1.0f;
    private int c1 = 1;
    private int c2 = 1;
    private int h1 = 1;
    private int h2 = 1;
    private int w1 = 1;
    private int w2 = 1;
    private int stride = 1;
    private int sample = 1;
    private int minh = 0;
    private int minw = 0;
    private int minc = 0;
    private int size = 0;

    public ShotcutKernel(int c1, int h1, int w1, int c2, int h2, int w2, CUDAManager cudaManager) {
        super(cudaManager);
        this.c1 = c1;
        this.c2 = c2;
        this.h1 = h1;
        this.h2 = h2;
        this.w1 = w1;
        this.w2 = w2;
        this.minw = (w1 < w2) ? w1 : w2;
        this.minh = (h1 < h2) ? h1 : h2;
        this.minc = (c1 < c2) ? c1 : c2;
        this.stride = w1 / w2;
        this.sample = w2 / w1;
        assert (stride == h1 / h2);
        assert (sample == h2 / h1);
        if (stride < 1) {
            stride = 1;
        }
        if (sample < 1) {
            sample = 1;
        }
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C1 = 6;
        int H1 = 4;
        int W1 = 4;
        int C2 = 3;
        int H2 = 8;
        int W2 = 8;
        float[] x1 = RandomUtils.order(N * C1 * H1 * W1, 0.1f, 0.1f);
        float[] x2 = RandomUtils.order(N * C2 * H2 * W2, 0.01f, 0.01f);
        float[] x3 = RandomUtils.order(N * C2 * H2 * W2, 0.01f, 0.01f);
        float[] d = RandomUtils.order(N * C2 * H2 * W2, 0.0001f, 0.0001f);
        Tensor input = new Tensor(N, C1, H1, W1, x1, true);
        Tensor output = new Tensor(N, C2, H2, W2, x2, true);
        Tensor output_cpu = new Tensor(N, C2, H2, W2, x3, true);
        Tensor delta = new Tensor(N, C2, H2, W2, d, true);
        CUDAManager cudaManager = new CUDAManager(0);
        ShotcutKernel k = new ShotcutKernel(C1, H1, W1, C2, H2, W2, cudaManager);
        ShotcutKernel k2 = new ShotcutKernel(C1, H1, W1, C2, H2, W2, cudaManager);
        //	    	output.showDM();
        k.shortcut(input, output);
        output.showDM();
        k2.shortcut_cpu(input, output_cpu);
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
                function = CUDAModules.getLocalFunctionByModule("ShortcutKernel.cu", "shortcut_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void shortcut(Tensor input, Tensor output) {
        try {
            if (kernelParameters == null || input.number != this.N) {
                this.size = input.number * minw * minh * minc;
                /**
                 * 设置入参
                 * int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out

                 */
                kernelParameters = Pointer.to(Pointer.to(new int[]{size}), Pointer.to(new int[]{minw}), Pointer.to(new int[]{minh}), Pointer.to(new int[]{minc}), Pointer.to(new int[]{stride}), Pointer.to(new int[]{sample}), Pointer.to(new int[]{input.number}), Pointer.to(new int[]{w1}), Pointer.to(new int[]{h1}), Pointer.to(new int[]{c1}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{w2}), Pointer.to(new int[]{h2}), Pointer.to(new int[]{c2}), Pointer.to(new float[]{s1}), Pointer.to(new float[]{s2}), Pointer.to(output.getGpuData()));
                this.N = output.number;
            }
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(size), 1, 1,      // Grid dimension
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

    public void shortcut_cpu(Tensor input, Tensor output) {
        int stride = input.width / output.width;
        int sample = output.width / input.width;
        if (stride < 1)
            stride = 1;
        if (sample < 1)
            sample = 1;
        int minw = (input.width < output.width) ? input.width : output.width;
        int minh = (input.height < output.height) ? input.height : output.height;
        int minc = (input.channel < output.channel) ? input.channel : output.channel;
        int i, j, k, b;
        for (b = 0; b < input.number; ++b) {
            for (k = 0; k < minc; ++k) {
                for (j = 0; j < minh; ++j) {
                    for (i = 0; i < minw; ++i) {
                        int out_index = i * sample + output.width * (j * sample + output.height * (k + output.channel * b));
                        int add_index = i * stride + input.width * (j * stride + input.height * (k + input.channel * b));
                        output.data[out_index] = s1 * output.data[out_index] + s2 * input.data[add_index];
                    }
                }
            }
        }
        System.out.println(JsonUtils.toJson(output.data));
    }
}

