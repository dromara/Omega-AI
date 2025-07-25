package com.omega.engine.nn.layer.dit.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class TokenDropKernel extends CUDAKernel {
    private CUfunction function;
    private CUfunction timestep_embedding_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public TokenDropKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "token_drop");
            }
            if (timestep_embedding_function == null) {
            	timestep_embedding_function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "timestep_embedding");
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
        int N = 2;
        int maxLen = 8;
        int headNum = 4;
        Tensor input = new Tensor(N, 1, maxLen, headNum, RandomUtils.order(N * maxLen * headNum, 0.1f, 0.1f), true);
        CUDAManager cudaManager = new CUDAManager(0);
        TokenDropKernel maskKernel = new TokenDropKernel(cudaManager);
        Tensor mask = new Tensor(N, 1, 1, 1, new float[] {0.05f, 0.4f}, true);
        Tensor param = new Tensor(1, 1, maxLen, headNum, RandomUtils.val(maxLen * headNum, 1.0f), true);
        input.showDM();
        maskKernel.tokenDrop(input, param, mask, input, 0.1f);
        input.showDM();
        PrintUtils.printImage(input);
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void tokenDrop(Tensor x, Tensor param, Tensor mask, Tensor output, float prob) {
        try {
            /**
             * 设置入参
             * const size_t size, const float *x, float *param, float *mask, float *out, const int len, float prob
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(param.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{x.getOnceSize()}), Pointer.to(new float[]{prob}));
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
    
    public void timestep_embedding(Tensor t, Tensor freqs, Tensor output, int d_model) {
        try {
            /**
             * 设置入参
             * int N, const float *t, float *freqs, float *out, const int d_model
             */
        	int N = t.dataLength * d_model / 2;
            kernelParameters = Pointer.to(Pointer.to(new long[]{N}), Pointer.to(t.getGpuData()), Pointer.to(freqs.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{d_model}));
            cuLaunchKernel(timestep_embedding_function, this.CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
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

}

