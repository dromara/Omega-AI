package com.omega.example.yolo.data;

import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class DataNormalization {
    public int N;
    public Tensor mean;
    public Tensor std;
    private CUfunction function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public DataNormalization(Tensor mean, Tensor std) {
        this.mean = mean;
        this.std = std;
        init();
    }

    public DataNormalization(float[] meanArray, float[] stdArray) {
        this.mean = new Tensor(1, 1, 1, 3, meanArray, true);
        this.std = new Tensor(1, 1, 1, 3, stdArray, true);
        init();
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
                function = CUDAModules.getLocalFunctionByModule("DataNormalization.cu", "normalization");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void normalization(Tensor input) {
        try {
            if (kernelParameters == null || this.N != input.number) {
                this.N = input.number;
                /**
                 * 设置入参
                 * float *input, float *mean, float *std, int N, int filters, int spatial

                 */
                kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(std.getGpuData()), Pointer.to(new int[]{input.dataLength}), Pointer.to(new int[]{input.channel}), Pointer.to(new int[]{input.height * input.width}));
            }
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

