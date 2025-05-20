package com.omega.engine.nn.layer.normalization.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * mean = batch均值    : 1/n∑xi
 * <p>
 * var = batch方差    : 1/n∑(xi - mean)^2
 * <p>
 * std = sqrt(var + eta)
 * <p>
 * xhati = (xi - mean) / std
 * <p>
 * yi = gama * xhati + beta
 * <p>
 * dgama = ∑delta * xhat
 * <p>
 * dbeta = ∑delta
 * <p>
 * dxhati = gama * deltai
 * <p>
 * dxi = 1 / std * (dxhati - mean(dxhat) - xhati * mean(dxhat * xhat))
 */
public class BNKernel3 extends BNBaseKernel {
    private int C;
    private int H;
    private int W;
    private int spatial;
    private CUfunction mwa_function;
    /**
     * 向前方法
     */
    private CUfunction fast_mean_function;
    private CUfunction fast_var_function;
    private CUfunction normalize_function;
    private CUfunction normalize_test_function;
    /**
     * 反向传播方法
     */
    private CUfunction dgamma_function;
    private CUfunction dbeta_function;
    private CUfunction dbeta_full_function;
    private CUfunction dxhat_function;
    private CUfunction fast_dmean_function;
    private CUfunction fast_dvar_function;
    private CUfunction dx_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private float eta = 1e-5f;
    private float momentum = 0.01f;
    /**
     * 前向参数
     */
    private CUdeviceptr d_z;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_var;
    /**
     * 反向参数
     */
    private CUdeviceptr d_dmean;
    private CUdeviceptr d_dvar;
    /**
     * 前向方法参数
     */
    private Pointer fastMeanParameters;
    private Pointer fastVarParameters;
    private Pointer normalizeParameters;
    private Pointer normalize_test_Parameters;
    private Pointer mwaParameters;
    /**
     * 反向方法参数
     */
    private Pointer dgammaParameters;
    private Pointer dbetaParameters;
    private Pointer dbetaFullParameters;
    private Pointer dxhatParameters;
    private Pointer fastDmeanParameters;
    private Pointer fastDvarParameters;
    private Pointer dxParameters;

    public BNKernel3(int C, int H, int W, Tensor runingMean, Tensor runingVar, CUDAManager cudaManager) {
        super(cudaManager);
        this.C = C;
        this.H = H;
        this.W = W;
        this.spatial = H * W;
        this.runingMean = runingMean;
        this.runingVar = runingVar;
        init();
    }

    public void initFunction() {
        try {
            if (fast_mean_function == null) {
                fast_mean_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "fast_mean_kernel");
            }
            if (fast_var_function == null) {
                fast_var_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "fast_variance_kernel");
            }
            if (normalize_function == null) {
                normalize_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "normalize_kernel");
            }
            if (normalize_test_function == null) {
                normalize_test_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "normalize_test_kernel");
            }
            if (mwa_function == null) {
                mwa_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "mwa_kernel");
            }
            if (dgamma_function == null) {
                dgamma_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "backward_scale_kernel");
            }
            if (dbeta_function == null) {
                dbeta_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "backward_bias_kernel");
            }
            if (dbeta_full_function == null) {
                dbeta_full_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "backward_bias_conn_kernel");
            }
            if (dxhat_function == null) {
                dxhat_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "scale_bias_kernel");
            }
            if (fast_dmean_function == null) {
                fast_dmean_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "fast_mean_delta_kernel");
            }
            if (fast_dvar_function == null) {
                fast_dvar_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "fast_variance_delta_kernel");
            }
            if (dx_function == null) {
                dx_function = getCudaManager().getLocalFunctionByModule("BNKernel3.cu", "normalize_delta_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    private void initKernel() {
        //		System.out.println(C);
        /**
         * 申请向前传播参数显存

         */
        this.d_mean = CUDAMemoryManager.getDevice(C);
        this.d_var = CUDAMemoryManager.getDevice(C);
        this.d_dmean = CUDAMemoryManager.getDevice(C);
        this.d_dvar = CUDAMemoryManager.getDevice(C);
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
        /**
         * 申请显存

         */
        initKernel();
    }

    public void initForward(Tensor input, Tensor gama, Tensor beta, Tensor output) {
        if (input.number != this.N) {
            this.N = input.number;
            /**
             * float *x, int batch, int filters, int spatial, float *mean

             */
            fastMeanParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_mean));
            /**
             * float *x, float *mean, int batch, int filters, int spatial, float *variance

             */
            fastVarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_var));
            /**
             * float* mean,float* var,float* runingMean,float* runingStd,int n

             */
            mwaParameters = Pointer.to(Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(runingMean.getGpuData()), Pointer.to(runingVar.getGpuData()), Pointer.to(new int[]{C}), Pointer.to(new float[]{momentum}));
            d_z = CUDAMemoryManager.getDevice(N * C * H * W);
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalizeParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(d_z), Pointer.to(output.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(gama.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(new float[]{eta}));
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalize_test_Parameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(runingMean.getGpuData()), Pointer.to(runingVar.getGpuData()), Pointer.to(gama.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(new float[]{eta}));
        }
    }

    public void initBackward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgama, Tensor dbeta) {
        if (dgammaParameters == null) {
            //			diff.setGpuData(delta.getGpuData());
            /**
             * float *x_norm, float *delta, int batch, int n, int size, float *scale_updates

             */
            dgammaParameters = Pointer.to(Pointer.to(d_z), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(dgama.getGpuData()));
            /**
             * float *bias_updates, float *delta, int batch, int n, int size

             */
            dbetaParameters = Pointer.to(Pointer.to(dbeta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}));
            /**
             * float *bias_updates, float *delta, int batch, int n

             */
            dbetaFullParameters = Pointer.to(Pointer.to(dbeta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}));
            /**
             * float *output, float *biases, int n, int size

             */
            dxhatParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}));
            /**
             * float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta

             */
            fastDmeanParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(d_dmean));
            /**
             * float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta

             */
            fastDvarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(d_dvar));
            /**
             * int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta

             */
            dxParameters = Pointer.to(Pointer.to(new int[]{N * C * spatial}), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_dmean), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(diff.getGpuData()));
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(RunModel RUN_MODEL, Tensor gama, Tensor beta, Tensor input, Tensor output) {
        initForward(input, gama, beta, output);
        if (RUN_MODEL == RunModel.TRAIN) {
            /**
             * 计算标准差
             * mean = 1/m ∑(x)
             * var = 1/m ∑(x - mean)^2
             * std = (var + eta)^1/2

             */
            fast_mean();
            fast_var();
            /**
             * 移动加权平均法计算均值与方差

             */
            mwa();
            normalize_train(input, gama, beta, output);
        } else {
            normalize_test(input, gama, beta, output);
        }
    }

    public void fast_mean() {
        try {
            cuLaunchKernel(fast_mean_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fastMeanParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fast_var() {
        try {
            cuLaunchKernel(fast_var_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fastVarParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void normalize_test(Tensor input, Tensor gama, Tensor beta, Tensor output) {
        try {
            cuLaunchKernel(normalize_test_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    normalize_test_Parameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void normalize_train(Tensor input, Tensor gama, Tensor beta, Tensor output) {
        try {
            cuLaunchKernel(normalize_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    normalizeParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mwa() {
        try {
            cuLaunchKernel(mwa_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    mwaParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        initBackward(input, delta, diff, gamma, dgamma, dbeta);
        if (spatial == 1) {
            dbetaFull();
        } else {
            dbeta();
            //			showDM(dbeta.getGpuData(), C);
        }
        dgamma();
        dxhat();
        fastDmean();
        fastDvar();
        //
        //		showDM(d_dvar, C);
        //		showDM(d_dmean, C);
        //
        dx();
        //		showDM(diff.getGpuData(), diff.dataLength);
    }

    public void dgamma() {
        try {
            cuLaunchKernel(dgamma_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    dgammaParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void dbeta() {
        try {
            cuLaunchKernel(dbeta_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    dbetaParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void dbetaFull() {
        try {
            cuLaunchKernel(dbeta_full_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    dbetaFullParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void dxhat() {
        try {
            cuLaunchKernel(dxhat_function, (spatial - 1) / CAFFE_CUDA_NUM_THREADS + 1, C, N,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    dxhatParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fastDmean() {
        try {
            cuLaunchKernel(fast_dmean_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fastDmeanParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fastDvar() {
        try {
            cuLaunchKernel(fast_dvar_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fastDvarParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void dx() {
        try {
            cuLaunchKernel(dx_function, this.CAFFE_GET_BLOCKS(N * C * spatial), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    dxParameters, null // Kernel- and extra parameters
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

    public void showDM(String id, CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
        System.out.println(id + ":" + JsonUtils.toJson(data));
    }
}

