package com.omega.engine.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.PrintUtils;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MaskKernel extends CUDAKernel {
    private CUfunction function;
    private CUfunction unmask_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public MaskKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int maxLen = 8;
        int headNum = 2;
        float[] x = new float[]{6, 8};
        Tensor input = new Tensor(N, 1, 1, 1, x, true);
        CUDAManager cudaManager = new CUDAManager(0);
        MaskKernel maskKernel = new MaskKernel(cudaManager);
        Tensor output = maskKernel.createOutput(N, maxLen, headNum);
        maskKernel.createHeadMask(input, output, N, maxLen, headNum);
        input.showDM();
        output.showDM();
        PrintUtils.printImage(output);
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("MaskKernel.cu", "createHeadMask");
            }
            if (unmask_function == null) {
                unmask_function = getCudaManager().getLocalFunctionByModule("MaskKernel.cu", "createHeadUnMask");
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

    public void createHeadMask(Tensor lens, Tensor mask, int number, int maxLen, int headNum) {
        try {
            /**
             * 设置入参
             *const size_t size, const float *lens, float *mask,const int number,const int maxLen,const headNum

             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{mask.dataLength}), Pointer.to(lens.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(new int[]{number}), Pointer.to(new int[]{maxLen}), Pointer.to(new int[]{headNum}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(mask.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void createHeadUnMask(Tensor lens, Tensor mask, int number, int max_label_len, int max_feat_len, int headNum) {
        try {
            /**
             * 设置入参
             *const size_t size, const float *lens, float *mask,const int number,const int max_label_len,const int max_feat_len,const int headNum

             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{mask.dataLength}), Pointer.to(lens.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(new int[]{number}), Pointer.to(new int[]{max_label_len}), Pointer.to(new int[]{max_feat_len}), Pointer.to(new int[]{headNum}));
            cuLaunchKernel(unmask_function, this.CAFFE_GET_BLOCKS(mask.dataLength), 1, 1,      // Grid dimension
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

    public Tensor createOutput(int number, int maxLen, int headNum) {
        return new Tensor(number, headNum, maxLen, maxLen, true);
    }

    public Tensor createUnMaskOutput(int number, int max_label_len, int max_feat_len, int headNum) {
        return new Tensor(number, headNum, max_label_len, max_feat_len, true);
    }
}

