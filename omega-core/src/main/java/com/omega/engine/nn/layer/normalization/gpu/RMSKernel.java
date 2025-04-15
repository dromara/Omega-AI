package com.omega.engine.nn.layer.normalization.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.utils.JsonUtils;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Transformer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Root Mean Sqrt Normalization
 * <p>
 * p = x^2
 * <p>
 * mean = 1/n∑x^2
 * <p>
 * rms = rsqrt(1/n∑x^2)
 * <p>
 * rms_norm = x * rms
 * <p>
 * drms = sum(x * diff)
 * <p>
 * dmean = -0.5 * (mean).pow(-1.5)
 * <p>
 * dp = 1/n
 * <p>
 * dx = rms * diff + sum(x * diff) * -0.5 * (mean).pow(-1.5) / n * 2 * x
 */
public class RMSKernel extends BaseKernel {
    public BNType bnType = null;
    private int B;
    private int W;
    /**
     * 向前方法
     */
    private CUfunction forward_function;
    private CUfunction forward_function2;
    private CUfunction forward_function3;
    /**
     * 反向传播方法
     */
    private CUfunction backward_function;
    private CUfunction backward_function2;
    private CUfunction backward_function3;
    private int CAFFE_CUDA_NUM_THREADS = 512;
    /**
     * 前向方法参数
     */
    private Pointer forwardParameters;
    private Pointer backwardParameters;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_rms;

    public RMSKernel(int W, BNType bnType, CUDAManager cudaManager) {
        super(cudaManager);
        this.W = W;
        this.bnType = bnType;
        init();
    }

    public static void main(String[] args) {
        try {
            int N = 4;
            int T = 512;
            int W = 512;
            float[] data = RandomUtils.order(N * T * W, 0.1f, 0.1f);
            Tensor input = new Tensor(N * T, 1, 1, W, data, true);
            Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.order(N * T * W, 0.1f, 0.1f), true);
            //    	    	Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.order(N * T * W, 0.1f, 0.1f), true);
            float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
            Tensor gamma = new Tensor(1, 1, 1, W, gammaData, true);
            Tensor dgamma = new Tensor(1, 1, 1, W, true);
            Transformer tf = new Transformer();
            //    			tf.number = N * T;
            RMSLayer rms = new RMSLayer(tf);
            rms.gamma = gamma;
            rms.diffGamma = dgamma;
            //	    	input.showDM();
            for (int i = 0; i < 10; i++) {
                rms.forward(input);
                rms.getOutput().showDMByNumber(0);
                rms.back(delta);
                rms.diff.showDMByNumber(0);
                rms.diffGamma.showDMByNumber(0);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    private void initKernel() {
        //		if(aten_mean == null || aten_mean.number != B) {
        /**
         * 申请向前传播参数显存

         */
        if (this.d_mean != null) {
            CUDAMemoryManager.free(this.d_mean);
            CUDAMemoryManager.free(this.d_rms);
        }
        //			System.out.println(B);
        this.d_mean = CUDAMemoryManager.getDevice(B);
        this.d_rms = CUDAMemoryManager.getDevice(B);
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_forward_kernel");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_backward_kernel");
            }
            if (forward_function2 == null) {
                forward_function2 = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_forward_kernel1");
            }
            if (backward_function2 == null) {
                backward_function2 = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_backward_kernel1");
            }
            if (forward_function3 == null) {
                forward_function3 = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_forward_kernel2");
            }
            if (backward_function3 == null) {
                backward_function3 = getCudaManager().getLocalFunctionByModule("RMSKernel.cu", "rmsnorm_backward_kernel2");
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

    public boolean checkBatch(Tensor input) {
        int batchSize = 0;
        switch (bnType) {
            case fully_bn:
                batchSize = input.number * input.channel * input.height;
                break;
            case conv_bn:
                batchSize = input.number * input.channel;
                break;
        }
        if (B != batchSize) {
            this.B = batchSize;
            return false;
        }
        return true;
    }

    public void initBackward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor gamma, Tensor input, Tensor output) {
        try {
            boolean check = checkBatch(input);
            if (!check) {
                initKernel();
            }
            /**
             * float* __restrict__ out, float* __restrict__ smean, float* __restrict__ rms,const float*  __restrict__ weight, const float*  __restrict__ inp, int N, int C

             */
            forwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(d_mean), Pointer.to(d_rms), Pointer.to(gamma.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            checkCUDA(cuLaunchKernel(forward_function, B, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward2(Tensor gamma, Tensor input, Tensor output) {
        try {
            checkBatch(input);
            /**
             * float *out, const float *inp, const float *weight, int N, int C

             */
            forwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            int grid_size = CAFFE_GET_BLOCKS(B);
            checkCUDA(cuLaunchKernel(forward_function2, grid_size, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward3(Tensor gamma, Tensor input, Tensor output) {
        try {
            checkBatch(input);
            /**
             * float *__restrict__ out, const float *__restrict__ inp,const float *__restrict__ weight, int N, int C

             */
            forwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            int grid_size = (int) Math.ceil((float) (B * 32 + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS);
            checkCUDA(cuLaunchKernel(forward_function3, grid_size, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma) {
        try {
            dgamma.clearGPU();
            /**
             * float* __restrict__ out, float* __restrict__ dweight, float* __restrict__ smean, float* __restrict__ rms, const float*  __restrict__ inp, const float*  __restrict__ delta, const float* __restrict__ weight, int N, int C

             */
            backwardParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dgamma.getGpuData()), Pointer.to(d_mean), Pointer.to(d_rms), Pointer.to(input.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            int shared_mem_size = (W + 1) * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(backward_function, B, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward2(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma) {
        try {
            dgamma.clearGPU();
            /**
             * float *dinp, float *dweight,const float *dout, const float *inp, const float *weight,int N, int C

             */
            backwardParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dgamma.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            int grid_size = CAFFE_GET_BLOCKS(B);
            checkCUDA(cuLaunchKernel(backward_function2, grid_size, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward3(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma) {
        try {
            dgamma.clearGPU();
            /**
             * float *__restrict__ dinp, float *__restrict__ dweight,

             const float *__restrict__ dout, const float *__restrict__ inp, const float *__restrict__ weight,

             int N, int C

             */
            backwardParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dgamma.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            int grid_size = (int) Math.ceil((float) (B * 32 + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS);
            //			System.out.println(grid_size);
            //			int grid_size = CAFFE_GET_BLOCKS(B);
            checkCUDA(cuLaunchKernel(backward_function3, grid_size, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
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

    public void showDM(String id, CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
        System.out.println(id + ":" + JsonUtils.toJson(data));
    }
}

