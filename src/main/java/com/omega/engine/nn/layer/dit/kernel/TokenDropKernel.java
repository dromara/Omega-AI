package com.omega.engine.nn.layer.dit.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class TokenDropKernel extends CUDAKernel {
	
    private CUfunction function;
    private CUfunction function2;
    private CUfunction timestep_embedding_function;
    private CUfunction img_token_drop_function;
    private CUfunction img_token_drop_back_function;
    
    private CUfunction rnd_int_function;
    
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
            if (function2 == null) {
            	function2 = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "token_drop_class");
            }
            if (timestep_embedding_function == null) {
            	timestep_embedding_function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "timestep_embedding");
            }
            if (img_token_drop_function == null) {
            	img_token_drop_function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "img_token_drop");
            }
            if (img_token_drop_back_function == null) {
            	img_token_drop_back_function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "img_token_drop_back");
            }
            if (rnd_int_function == null) {
            	rnd_int_function = getCudaManager().getLocalFunctionByModule("TokenDropKernel.cu", "generateRandomUniqueIntegers");
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
//        int N = 2;
//        int maxLen = 8;
//        int headNum = 4;
//        Tensor input = new Tensor(N, 1, maxLen, headNum, RandomUtils.order(N * maxLen * headNum, 0.1f, 0.1f), true);
//        CUDAManager cudaManager = new CUDAManager(0);
//        TokenDropKernel maskKernel = new TokenDropKernel(cudaManager);
//        Tensor mask = new Tensor(N, 1, 1, 1, new float[] {0.05f, 0.4f}, true);
//        Tensor param = new Tensor(1, 1, maxLen, headNum, RandomUtils.val(maxLen * headNum, 1.0f), true);
//        input.showDM();
//        maskKernel.tokenDrop(input, param, mask, input, 0.1f);
//        input.showDM();
//        PrintUtils.printImage(input);
    	
    	int N = 32;
    	int T = 256;
    	int W = 64;
    	Tensor idsKeep = new Tensor(N, 1, 1, W, true);
    	CUDAManager cudaManager = new CUDAManager(0);
    	TokenDropKernel maskKernel = new TokenDropKernel(cudaManager);
    	maskKernel.idsKeep(idsKeep, N, T - 1, W);
    	idsKeep.showDM("idsKeep");
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void tokenDrop(Tensor x, Tensor param, Tensor mask, Tensor output, int W, float prob) {
        try {
      
            /**
             * 设置入参
             * const size_t size, const float *x, float *param, float *mask, float *out, const int len, float prob
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(param.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{W}), Pointer.to(new float[]{prob}));
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
    
    public void tokenDrop(Tensor x, float param, Tensor mask, Tensor output, float prob) {
        try {
            /**
             * 设置入参
             * const size_t size, const float *x, float *param, float *mask, float *out, const int len, float prob
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(new float[] {param}), Pointer.to(mask.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{x.getOnceSize()}), Pointer.to(new float[]{prob}));
            cuLaunchKernel(function2, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void idsKeep(Tensor idsKeep, int batch, int T, int N) {
        try {
        	int seed = RandomUtils.rand();

            /**
             * 设置入参
             * int B, int N, int T, float *output, unsigned int seed
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(new int[]{batch}), Pointer.to(new int[]{N}), Pointer.to(new int[]{T}), Pointer.to(idsKeep.getGpuData()), Pointer.to(new int[]{seed}));
            cuLaunchKernel(rnd_int_function, batch, 1, 1,      // Grid dimension
                    1, 1, 1,      // Block dimension
                    N * Sizeof.INT, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void imgTokenDrop(Tensor x, Tensor idskeep, Tensor output, int xT, int T, int W) {
        try {
            /**
             * 设置入参
             * const size_t size, const float *x, float *idskeep, float *out, const int xT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{output.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{xT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(img_token_drop_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void imgTokenDropBack(Tensor dx, Tensor idskeep, Tensor delta, int xT, int T, int W) {
        try {
            /**
             * 设置入参
             * const size_t size, float *dx, float *idskeep, const float *dout, const int xT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{delta.dataLength}), Pointer.to(dx.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{xT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(img_token_drop_back_function, this.CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
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

