package com.omega.engine.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.utils.CheckArrayUtils;
import com.omega.utils.JsonUtils;
import com.omega.utils.MatrixOperation;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.MWAUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import java.util.Vector;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class BNKernel extends CUDAKernel {
    private String id;
    private BNType bnType;
    private Tensor x;
    private float[] gama;
    private float[] beta;
    private Tensor out;
    private Tensor delta;
    private float[] dgama;
    private float[] dbeta;
    private Tensor diff;
    private int N = 0;
    private int C;
    private int H;
    private int W;
    private float scale;
    private CUfunction mean_function;
    private CUfunction var_function;
    private CUfunction std_function;
    private CUfunction mwa_function;
    private CUfunction culOutput_function;
    /**
     * fast function
     */
    private CUfunction fast_mean_function;
    private CUfunction fast_var_function;
    private CUfunction normalize_function;
    private CUfunction computeDiff_function;
    private CUfunction computeDelta_function;
    private CUfunction computeDelta_full_function;
    private CUfunction meanDzSum_function;
    /**
     * fast function
     */
    private CUfunction dgama_function;
    private CUfunction dbeta_function;
    private CUfunction dxhat_function;
    private CUfunction full_dmean_function;
    private CUfunction full_dvar_function;
    private CUfunction fast_dmean_function;
    private CUfunction fast_dvar_function;
    private CUfunction dx_function;
    private CUfunction dx_full_function;
    //	private CUfunction computeDgama_function;
    //	private CUfunction meanDelta_function;
    //	private CUfunction dx_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    /**
     * 前向参数
     */
    private CUdeviceptr d_x;
    private CUdeviceptr d_gama;
    private CUdeviceptr d_beta;
    private CUdeviceptr d_z;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_var;
    private CUdeviceptr d_std;
    private CUdeviceptr d_runingMean;
    private CUdeviceptr d_runingVar;
    private CUdeviceptr d_out;
    /**
     * 反向参数
     */
    private CUdeviceptr d_delta;
    private CUdeviceptr d_dgama;
    private CUdeviceptr d_dbeta;
    private CUdeviceptr d_dmean;
    private CUdeviceptr d_dvar;
    private CUdeviceptr d_dmu;
    private CUdeviceptr d_diff;
    /**
     * 前向方法
     */
    private Pointer meanParameters;
    private Pointer varParameters;
    private Pointer fullDmeanParameters;
    private Pointer fullDvarParameters;
    private Pointer fastMeanParameters;
    private Pointer fastVarParameters;
    private Pointer normalizeParameters;
    private Pointer mwaParameters;
    /**
     * 反向方法

     */
    /**
     * fast function
     */
    private Pointer dgamaParameters;
    private Pointer dbetaParameters;
    private Pointer computeDelta_full_Parameters;
    private Pointer dxhatParameters;
    private Pointer fastDmeanParameters;
    private Pointer fastDvarParameters;
    private Pointer dxParameters;
    private Pointer dx_fullParameters;
    private float eta = 0.00001f;
    private float[] rm;
    private float[] rv;

    public BNKernel(String id, BNType bnType, Tensor out, Tensor diff, float[] dgama, float[] dbeta, int N, int C, int H, int W, CUDAManager cudaManager) {
        super(cudaManager);
        this.id = id;
        this.bnType = bnType;
        this.N = N;
        this.C = C;
        this.H = H;
        this.W = W;
        this.out = out;
        this.diff = diff;
        this.dgama = dgama;
        this.dbeta = dbeta;
        this.scale = 1.0f / (N * H * W);
        init();
    }

    public void initFunction() {
        try {
            if (mean_function == null) {
                mean_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "mean_cov");
            }
            if (fast_mean_function == null) {
                fast_mean_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "fast_mean_kernel");
            }
            if (var_function == null) {
                var_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "var_cov");
            }
            if (fast_var_function == null) {
                fast_var_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "fast_variance_kernel");
            }
            if (normalize_function == null) {
                normalize_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "normalize_kernel");
            }
            if (std_function == null) {
                std_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "std_fn");
            }
            if (mwa_function == null) {
                mwa_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "mwa");
            }
            if (culOutput_function == null) {
                culOutput_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "culOutput_cov");
            }
            if (computeDelta_function == null) {
                computeDelta_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "computeDelta");
            }
            if (computeDelta_full_function == null) {
                computeDelta_full_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "computeDelta_full");
            }
            if (meanDzSum_function == null) {
                meanDzSum_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "meanDzSum");
            }
            if (computeDiff_function == null) {
                computeDiff_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "computeDiff");
            }
            /**
             * fast function

             */
            if (dgama_function == null) {
                dgama_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "dgama_kernel");
            }
            if (dbeta_function == null) {
                dbeta_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "dbeta_kernel");
            }
            if (dxhat_function == null) {
                dxhat_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "dxhat_kernel2");
            }
            if (full_dmean_function == null) {
                full_dmean_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "full_mean_delta_kernel");
            }
            if (fast_dmean_function == null) {
                fast_dmean_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "fast_mean_delta_kernel");
            }
            if (full_dvar_function == null) {
                full_dvar_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "full_var_delta_kernel");
            }
            if (fast_dvar_function == null) {
                fast_dvar_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "fast_variance_delta_kernel");
            }
            if (dx_function == null) {
                dx_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "dx_kernel");
            }
            if (dx_full_function == null) {
                dx_full_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "dx_kernel_full");
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
        /**
         * 申请内存

         */
        initKernel();
        /**
         * 申请显存

         */
        initInputDevice();
        initForward();
        initBackward();
    }

    public void initForward() {
        /**
         * float* x,float* mean,int number,int channel,int height,int width

         */
        meanParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
        /**
         * float* x,float* mean,float* var,int number,int channel,int height,int width

         */
        varParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
        /**
         * float *x, int batch, int filters, int spatial, float *mean

         */
        fastMeanParameters = Pointer.to(Pointer.to(d_x), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_mean));
        /**
         * float *x, float *mean, int batch, int filters, int spatial, float *variance

         */
        fastVarParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_var));
        /**
         * float* mean,float* var,float* runingMean,float* runingStd,int n

         */
        mwaParameters = Pointer.to(Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_runingMean), Pointer.to(d_runingVar), Pointer.to(new int[]{C}));
    }

    public void initBackward() {
        /**
         * fast function

         */
        /**
         * 设置入参
         * float *x_norm, float *delta, int batch, int c, int size, float *dgama

         */
        dgamaParameters = Pointer.to(Pointer.to(d_z), Pointer.to(d_delta), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dgama));
        /**
         * 设置入参
         * float *dbeta, float *delta, int batch, int c, int size

         */
        dbetaParameters = Pointer.to(Pointer.to(d_dbeta), Pointer.to(d_delta), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
        /**
         * 设置入参
         * float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int channel,int height,int width

         */
        computeDelta_full_Parameters = Pointer.to(Pointer.to(d_delta), Pointer.to(d_dgama), Pointer.to(d_dbeta), Pointer.to(d_z), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
        /**
         * 设置入参
         * float *dxhat, float *variance, int batch, int filters, float *mean_delta

         */
        fullDmeanParameters = Pointer.to(Pointer.to(d_diff), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(d_dmean));
        /**
         * 设置入参
         * float *dxhat, float *variance, int batch, int filters, int spatial, float *mean_delta

         */
        fastDmeanParameters = Pointer.to(Pointer.to(d_diff), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dmean));
        /**
         * 设置入参
         * float *x, float *dxhat, float *mean, float *variance, int batch, int filters, float *variance_delta

         */
        fullDvarParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_diff), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(d_dvar));
        /**
         * 设置入参
         * float *x, float *dxhat, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta

         */
        fastDvarParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_diff), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dvar));
    }

    private void initInputDevice() {
        this.scale = 1.0f / (N * H * W);
        this.d_x = CUDAMemoryManager.getDevice(N * C * H * W);
        this.d_z = CUDAMemoryManager.getDevice(N * C * H * W);
        this.d_out = CUDAMemoryManager.getDevice(N * C * H * W);
        this.d_diff = CUDAMemoryManager.getDevice(N * C * H * W);
        this.d_delta = CUDAMemoryManager.getDevice(N * C * H * W);
    }

    private void initKernel() {
        /**
         * 申请向前传播参数显存

         */
        this.d_gama = CUDAMemoryManager.getDevice(C);
        this.d_beta = CUDAMemoryManager.getDevice(C);
        this.d_mean = CUDAMemoryManager.getDevice(C);
        this.d_var = CUDAMemoryManager.getDevice(C);
        this.d_std = CUDAMemoryManager.getDevice(C);
        this.d_runingMean = CUDAMemoryManager.getDevice(C);
        this.d_runingVar = CUDAMemoryManager.getDevice(C);
        /**
         * 申请反向传播参数显存

         */
        this.d_dmean = CUDAMemoryManager.getDevice(C);
        this.d_dvar = CUDAMemoryManager.getDevice(C);
        this.d_dmu = CUDAMemoryManager.getDevice(C);
        this.d_dgama = CUDAMemoryManager.getDevice(C);
        this.d_dbeta = CUDAMemoryManager.getDevice(C);
    }

    public void setX(Tensor x, int number) {
        this.x = x;
        /**
         * 重新申请显存

         */
        if (number != N) {
            this.N = number;
            JCuda.cudaFree(d_x);
            JCuda.cudaFree(d_z);
            JCuda.cudaFree(d_out);
            JCuda.cudaFree(d_diff);
            JCuda.cudaFree(d_delta);
            /**
             * 申请内存

             */
            initInputDevice();
        }
        /**
         * 拷贝数据到显存

         */
        JCudaDriver.cuMemcpyHtoD(d_x, Pointer.to(x.data), x.data.length * Sizeof.FLOAT);
    }

    public void setDelta(Tensor delta) {
        this.delta = delta;
        /**
         * 拷贝数据到显存

         */
        JCudaDriver.cuMemcpyHtoD(d_delta, Pointer.to(delta.data), delta.data.length * Sizeof.FLOAT);
    }

    public void setGama(float[] gama, float[] beta) {
        this.gama = gama;
        this.beta = beta;
        /**
         * 拷贝数据到显存

         */
        JCudaDriver.cuMemcpyHtoD(d_gama, Pointer.to(gama), gama.length * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoD(d_beta, Pointer.to(beta), beta.length * Sizeof.FLOAT);
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(RunModel RUN_MODEL) {
        //		long start = System.nanoTime();
        if (RUN_MODEL == RunModel.TRAIN) {
            /**
             * 计算标准差
             * mean = 1/m ∑(x)
             * var = 1/m ∑(x - mean)^2
             * std = (var + eta)^1/2

             */
            if (bnType == BNType.fully_bn) {
                mean();
                var();
            } else {
                fast_mean();
                fast_var();
            }
            /**
             * 移动加权平均法计算均值与方差

             */
            mwa();
            normalize(d_mean, d_var);
        } else {
            normalize(d_runingMean, d_runingVar);
        }
    }

    public void mean() {
        try {
            cuLaunchKernel(mean_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    meanParameters, null // Kernel- and extra parameters
            );
            JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void var() {
        try {
            cuLaunchKernel(var_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    varParameters, null // Kernel- and extra parameters
            );
            JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fast_mean() {
        try {
            cuLaunchKernel(fast_mean_function, C, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fastMeanParameters, null // Kernel- and extra parameters
            );
            JCudaDriver.cuCtxSynchronize();
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
            JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void normalize(CUdeviceptr mean, CUdeviceptr var) {
        try {
            //			float[] m = new float[C];
            //			float[] v = new float[C];
            //
            //			showDM("", mean, m);
            //
            //			showDM("", var, v);
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalizeParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(d_x), Pointer.to(d_z), Pointer.to(d_out), Pointer.to(mean), Pointer.to(var), Pointer.to(d_gama), Pointer.to(d_beta), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
            cuLaunchKernel(normalize_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    normalizeParameters, null // Kernel- and extra parameters
            );
            JCudaDriver.cuMemcpyDtoH(Pointer.to(out.data), d_out, out.data.length * Sizeof.FLOAT);
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
            JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mwa_cpu() {
        try {
            if (rm == null || rv == null) {
                rm = new float[C];
                rv = new float[C];
            }
            float[] t = new float[C];
            float[] t2 = new float[C];
            getDM(d_mean, t);
            getDM(d_var, t2);
            MWAUtils.mwa(t, rm);
            MWAUtils.mwa(t2, rv);
            System.out.println(id + ":" + JsonUtils.toJson(rv));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward() {
        //		long start = System.nanoTime();
        if (bnType == BNType.fully_bn) {
            computeDelta_full();
        } else {
            computeDgama();
            computeDbeta();
        }
        //		System.out.println((System.nanoTime() - start) / 1e6 + "ms.1");
        //		long start2 = System.nanoTime();
        computeDxhat();
        if (bnType == BNType.fully_bn) {
            computeFullDmean();
            computeFullDvar();
        } else {
            computeDmean();
            computeDvar();
        }
        //		System.out.println((System.nanoTime() - start2) / 1e6 + "ms.2");
        //		long start3 = System.nanoTime();
        if (bnType == BNType.fully_bn) {
            computeDx_full();
        } else {
            computeDx();
        }
        //		System.out.println((System.nanoTime() - start3) / 1e6 + "ms.3");
        //
        //		System.out.println("===========>"+id);
        //		System.out.println((System.nanoTime() - start)/1e6+"ms.backward");
    }

    private void computeDgama() {
        cuLaunchKernel(dgama_function, C, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dgamaParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuMemcpyDtoH(Pointer.to(dgama), d_dgama, dgama.length * Sizeof.FLOAT);
    }

    private void computeDbeta() {
        cuLaunchKernel(dbeta_function, C, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dbetaParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuMemcpyDtoH(Pointer.to(dbeta), d_dbeta, dbeta.length * Sizeof.FLOAT);
    }

    private void computeDelta_full() {
        cuLaunchKernel(computeDelta_full_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                computeDelta_full_Parameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
        JCudaDriver.cuMemcpyDtoH(Pointer.to(dgama), d_dgama, dgama.length * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyDtoH(Pointer.to(dbeta), d_dbeta, dbeta.length * Sizeof.FLOAT);
    }

    private void computeDxhat() {
        /**
         * 设置入参
         * int N, float *delta, float *dz, float *gama, int filters, int spatial

         */
        dxhatParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(d_delta), Pointer.to(d_diff), Pointer.to(d_gama), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
        cuLaunchKernel(dxhat_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dxhatParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private void computeDmean() {
        cuLaunchKernel(fast_dmean_function, C, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fastDmeanParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private void computeFullDmean() {
        cuLaunchKernel(full_dmean_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fullDmeanParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private void computeDvar() {
        cuLaunchKernel(fast_dvar_function, C, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fastDvarParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private void computeFullDvar() {
        cuLaunchKernel(full_dvar_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fullDvarParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private void computeDx() {
        /**
         * 设置入参
         * int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *diff

         */
        dxParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(d_x), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_dmean), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_diff));
        cuLaunchKernel(dx_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dxParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuMemcpyDtoH(Pointer.to(diff.data), d_diff, diff.data.length * Sizeof.FLOAT);
    }

    private void computeDx_full() {
        /**
         * 设置入参
         * int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *diff

         */
        dx_fullParameters = Pointer.to(Pointer.to(d_x), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_dmean), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(d_diff));
        cuLaunchKernel(dx_full_function, this.CAFFE_GET_BLOCKS(C), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dx_fullParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuMemcpyDtoH(Pointer.to(diff.data), d_diff, diff.data.length * Sizeof.FLOAT);
    }

    public void free() {
        JCuda.cudaFree(d_x);
        JCuda.cudaFree(d_z);
        JCuda.cudaFree(d_out);
        JCuda.cudaFree(d_gama);
        JCuda.cudaFree(d_beta);
        JCuda.cudaFree(d_var);
        JCuda.cudaFree(d_mean);
        JCuda.cudaFree(d_std);
        JCuda.cudaFree(d_runingMean);
        JCuda.cudaFree(d_runingVar);
        JCuda.cudaFree(d_dmean);
        JCuda.cudaFree(d_dvar);
        JCuda.cudaFree(d_dmu);
        JCuda.cudaFree(d_diff);
        JCuda.cudaFree(d_delta);
        JCuda.cudaFree(d_dgama);
        JCuda.cudaFree(d_dbeta);
    }

    public Tensor getOut() {
        return out;
    }

    public Tensor getDiff() {
        return diff;
    }

    public float[] getDgama() {
        return dgama;
    }

    public float[] getDbeta() {
        return dbeta;
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    //    public static void main(String args[]){
    //    	int N = 128;
    //    	int C = 4096;
    //    	int H = 1;
    //    	int W = 1;
    //
    //    	float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.01f);
    //
    ////    	float[] x = MatrixUtils.one(N * C * H * W);
    //
    //    	float[] gama = RandomUtils.gaussianRandom(C, 0.1f);
    //
    //    	float[] beta = RandomUtils.gaussianRandom(C, 0.1f);
    //
    //    	float[] out = new float[N * C * H * W];
    //
    //    	float[] diff = new float[N * C * H * W];
    //
    //    	float[] dgama = new float[C];
    //
    //    	float[] dbeta = new float[C];
    //
    //    	float[][][][] x_cpu = MatrixUtils.transform(x, N, C, H, W);
    //
    //    	float[][][][] out_cpu = new float[N][C][H][W];
    //
    //    	float[][][][] diff_cpu = new float[N][C][H][W];
    //
    //    	BNKernel bn = new BNKernel("test",out, diff, dgama, dbeta, N, C, H, W);
    //
    //    	for(int i = 0;i<10;i++) {
    //
    //        	long start = System.nanoTime();
    //
    //        	bn.setX(x, N);
    //
    //        	bn.setGama(gama, beta);
    //
    //        	bn.forward(RunModel.TRAIN);
    //
    //        	bn.setDelta(x);
    //
    //        	bn.backward();
    //
    //        	System.out.println((System.nanoTime() - start) / 1e6+"ms.count");
    //    	}
    //
    //    	bn.foward_cpu(x_cpu, out_cpu, x_cpu, diff_cpu);
    //
    ////    	System.out.println(JsonUtils.toJson(x));
    //
    ////    	System.out.println(JsonUtils.toJson(bn.getOut()));
    ////
    ////    	System.out.println(JsonUtils.toJson(MatrixUtils.transform(out_cpu)));
    //
    //    	System.out.println("out error:"+CheckArrayUtils.oneCheck(MatrixUtils.transform(out_cpu), bn.getOut()));
    //
    ////    	float[] mean = new float[C];
    ////    	float[] var = new float[C];
    ////    	float[] z = new float[N * C * H *W];
    //
    ////    	bn.showDM("var_gpu",bn.d_var, var);
    ////    	bn.showDM("dmean_gpu",bn.d_dmean, mean);
    ////    	bn.showDM(bn.d_out, z);
    ////    	bn.showDM(bn.d_dmu, mean);
    //
    ////    	System.out.println(JsonUtils.toJson(bn.getDiff()));
    ////
    ////    	System.out.println(JsonUtils.toJson(MatrixUtils.transform(diff_cpu)));
    //
    //    	System.out.println("diff error:"+CheckArrayUtils.oneCheck(MatrixUtils.transform(diff_cpu), bn.getDiff()));
    //
    ////    	System.out.println(JsonUtils.toJson(bn.getDiff()));
    //
    //    	bn.free();
    //
    //    }
    public void showDM(String id, CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
        System.out.println(id + ":" + JsonUtils.toJson(data));
    }

    public void getDM(CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
    }

    public float[] getRuningMean() {
        if (rm == null) {
            rm = new float[C];
        }
        JCudaDriver.cuMemcpyDtoH(Pointer.to(rm), d_runingMean, rm.length * Sizeof.FLOAT);
        return rm;
    }

    public float[] getRuningVar() {
        if (rv == null) {
            rv = new float[C];
        }
        JCudaDriver.cuMemcpyDtoH(Pointer.to(rv), d_runingVar, rv.length * Sizeof.FLOAT);
        return rv;
    }

    public float[] setRuningMean(float[] rm) {
        JCudaDriver.cuMemcpyHtoD(d_runingMean, Pointer.to(rm), rm.length * Sizeof.FLOAT);
        return rm;
    }

    public float[] setRuningVar(float[] rv) {
        JCudaDriver.cuMemcpyHtoD(d_runingVar, Pointer.to(rv), rv.length * Sizeof.FLOAT);
        return rm;
    }

    public void foward_cpu(float[][][][] x, float[][][][] out, float[][][][] delta, float[][][][] diff) {
        float[] mean = new float[C];
        float[] var = new float[C];
        MatrixOperation.meanV2(x, mean, 1);
        /**
         * 计算标准差
         * var = 1/m ∑(x - mean)^2
         * std = (var + eta)^1/2

         */
        MatrixOperation.varV2(x, mean, var, 1);
        /**
         * zi = (xi - mean) / (std + eta)

         */
        float[][][][] z = this.culOutput(x, out, mean, var);
        float[] dvar = new float[C];
        float[] dmu = new float[C];
        float[] dgama = new float[C];
        float[] dbeta = new float[C];
        computeDelta_cpu(delta, z, gama, dgama, dbeta, diff);
        /**
         * 原论文公式
         * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
         * 使用darknet公式
         * dmean = (∑ dxhat * -1 / (var + eta)^1/2)

         */
        meanDzSum_cpu(dvar, mean, var, dmu, x, diff);
        //		System.out.println("dgama:"+JsonUtils.toJson(dgama));
        //		System.out.println("dbeta:"+JsonUtils.toJson(dbeta));
        System.out.println("dgama error:" + CheckArrayUtils.oneCheck(dgama, this.dgama));
        System.out.println("dbeta error:" + CheckArrayUtils.oneCheck(dbeta, this.dbeta));
        //float[][][][] diff,float[][][][] x,float[] mean,float[] dmu,float[] std,float[] dvar
        computeDiff_cpu(diff, x, mean, var, dmu, dvar);
        //		float[][][][] delta,float[][][][] z,float[][][][] diff,float[] std,float[] gama,float[] dgama,float[] dbeta
        //		backward_caffe(delta, z, diff, std, gama, dgama, dbeta);
    }

    private void computeDelta_cpu(float[][][][] delta, float[][][][] z, float[] gama, float[] dgama, float[] dbeta, float[][][][] diff) {
        for (int c = 0; c < C; c++) {
            dgama[c] = 0;
            dbeta[c] = 0;
            for (int m = 0; m < N; m++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        // deltaGama = ∑ deta * z
                        dgama[c] += delta[m][c][h][w] * z[m][c][h][w];
                        // deltaBeta = ∑ deta
                        dbeta[c] += delta[m][c][h][w];
                        // dxhat = deta * gama
                        diff[m][c][h][w] = delta[m][c][h][w] * gama[c];
                    }
                }
            }
        }
    }

    /**
     * 原论文公式
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
     * <p>
     * 使用darknet公式
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
     * <p>
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     */
    private void meanDzSum_cpu(float[] dvar, float[] mean, float[] var, float[] dmu, float[][][][] x, float[][][][] dz) {
        for (int c = 0; c < C; c++) {
            float dvar_val = 0.0f;
            float dmu_val = 0.0f;
            //			float dmu2_val = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dvar_val += (x[n][c][h][w] - mean[c]) * dz[n][c][h][w];
                        dmu_val += -1.0f * dz[n][c][h][w] / (float) Math.sqrt(var[c] + eta);
                        //						dmu2_val += -2.0f * (x[n][c][h][w] - mean[c]) * scale;
                    }
                }
            }
            dvar[c] = (float) (dvar_val * -0.5f * Math.pow(var[c] + eta, -1.5));
            //			dmu[c] = dmu_val + dmu2_val * dvar[c];
            dmu[c] = dmu_val;
        }
        //    	System.out.println("var_cpu:"+JsonUtils.toJson(var));
        //    	System.out.println("dmean_cpu:"+JsonUtils.toJson(dmu));
    }

    private void computeDiff_cpu(float[][][][] diff, float[][][][] x, float[] mean, float[] var, float[] dmu, float[] dvar) {
        for (int m = 0; m < N; m++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        // dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
                        diff[m][c][h][w] = diff[m][c][h][w] / (float) Math.sqrt(var[c] + eta) + 2.0f * dvar[c] * (x[m][c][h][w] - mean[c]) * scale + dmu[c] * scale;
                    }
                }
            }
        }
    }

    private float[][][][] culOutput(float[][][][] x, float[][][][] out, float[] m, float[] var) {
        int N = x.length;
        int C = x[0].length;
        int H = x[0][0].length;
        int W = x[0][0][0].length;
        float[][][][] z = new float[N][C][H][W];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < N; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                z[index][c][h][w] = (x[index][c][h][w] - m[c]) / (float) Math.sqrt(var[c] + eta);
                                out[index][c][h][w] = z[index][c][h][w] * gama[c] + beta[c];
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(8).dispatchTask(workers);
        return z;
    }
}

