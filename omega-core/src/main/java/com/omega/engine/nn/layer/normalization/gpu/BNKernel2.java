package com.omega.engine.nn.layer.normalization.gpu;

import com.omega.common.utils.*;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.layer.normalization.BNType;
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
public class BNKernel2 extends BNBaseKernel {
    private BNType bnType;
    private int C;
    private int H;
    private int W;
    private int meanNumber = 0;
    private CUfunction mean_function;
    private CUfunction var_function;
    private CUfunction mwa_function;
    private CUfunction culOutput_function;
    /**
     * fast function
     */
    private CUfunction fast_mean_function;
    private CUfunction fast_var_function;
    private CUfunction normalize_function;
    private CUfunction normalize_test_function;
    private CUfunction computeDiff_function;
    private CUfunction computeDelta_full_function;
    private CUfunction fast_mean_xhat_function;
    private CUfunction fast_mean_dxhat_function;
    private CUfunction mean_xhat_function;
    private CUfunction meanDzSum_function;
    /**
     * fast function
     */
    private CUfunction dgama_function;
    private CUfunction dbeta_function;
    private CUfunction dxhat_function;
    private CUfunction dx_function;
    private CUfunction dx_full_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private float eta = 1e-5f;
    private float momentum = 0.01f;
    /**
     * 前向参数
     */
    private CUdeviceptr d_z;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_var;
    private CUdeviceptr d_std;
    private CUdeviceptr d_runingMean;
    private CUdeviceptr d_runingVar;
    /**
     * 反向参数
     */
    private CUdeviceptr d_mean_dz;
    private CUdeviceptr d_mean_dzxz;
    /**
     * 前向方法
     */
    private Pointer meanParameters;
    private Pointer varParameters;
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
    private Pointer normalize_test_Parameters;
    private Pointer dxhatParameters;
    private Pointer dxParameters;
    private Pointer fast_mean_xhat_Parameters;
    private Pointer fast_mean_dxhat_Parameters;
    private Pointer mean_xhat_Parameters;
    private Pointer dx_fullParameters;

    public BNKernel2(BNType bnType, int C, int H, int W, CUDAManager cudaManager) {
        super(cudaManager);
        this.bnType = bnType;
        this.C = C;
        this.H = H;
        this.W = W;
        if (this.bnType == BNType.fully_bn) {
            meanNumber = W;
        } else {
            meanNumber = C;
        }
        init();
    }

    public static void main(String args[]) {
        test2d();
    }

    public static void test2d() {
        int N = 2;
        int C = 3;
        int H = 5;
        int W = 5;
        //    	float[] x = RandomUtils.order(N * C * H * W, 1.0f, 1f);
        float[] x = new float[]{-0.5596f, 0.6154f, -1.1204f, -0.1636f, -1.3229f, 0.9092f, -0.8235f, 0.3563f, 1.2746f, 0.6454f, -0.7569f, -0.6933f, -1.0571f, -1.5361f, 1.8608f, 0.4835f, -1.3317f, -2.3606f, 0.8470f, 1.1632f, 0.5080f, -1.4968f, 1.4136f, 0.8903f, 0.4200f, 0.8233f, -0.6349f, 0.4416f, 0.5081f, 0.1545f, 0.3967f, 0.6221f, -0.0245f, 0.8493f, -0.8964f, -0.5109f, -0.7737f, -0.2255f, 1.6705f, 0.2935f, 0.5887f, -1.7415f, 0.6597f, 1.2048f, 0.7282f, 0.8854f, 0.9372f, -0.0824f, 1.6266f, -1.9450f, -1.3224f, 0.5002f, -1.0779f, 0.9101f, -0.8541f, -0.5130f, -0.4204f, 0.1571f, -0.3905f, -1.3593f, 1.0415f, -0.9938f, 2.6900f, -1.1995f, 0.7727f, -0.2714f, 1.1784f, -0.8269f, 0.3220f, 0.7001f, -0.0134f, -1.1899f, -0.8730f, 1.1819f, 0.9492f, -1.3036f, 0.3672f, -0.1123f, -0.1105f, 0.4664f, -0.7766f, -0.1695f, -1.9371f, -0.4164f, 1.8046f, -0.0946f, 0.8305f, 0.9820f, 0.5660f, -0.1472f, 0.4830f, 1.0134f, 1.3013f, -1.4215f, 0.4570f, 1.5848f, -0.2974f, -1.2160f, 0.6511f, 0.4922f, -1.2987f, -0.9202f, -1.6065f, 0.6146f, -1.7012f, -0.5577f, 1.3336f, -0.5391f, 0.1539f, -0.7145f, 0.2365f, 1.0505f, 0.2315f, -1.4901f, 0.1007f, 0.7942f, -1.1326f, -1.6860f, -0.0734f, 0.9499f, 0.2508f, 1.3307f, -0.9660f, -1.3506f, -1.4267f, 1.1793f, -0.3751f, 0.7723f, -0.2359f, -0.2686f, 1.2551f, -0.6165f, -1.1625f, 0.5438f, -1.5241f, -1.8666f, -0.0040f, 1.6220f, -0.9495f, -0.8307f, -2.1322f, 0.3769f, 0.9336f, 0.2458f, 0.1653f, -0.4835f, -0.7139f, -0.4338f, 0.1007f, 1.4633f};
        float[] d = RandomUtils.val(N * C * H * W, 1.0f);
        float[] g = MatrixUtils.one(C);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor gama = new Tensor(1, 1, 1, C, g, true);
        Tensor beta = new Tensor(1, 1, 1, C, true);
        Tensor delta = new Tensor(N, C, H, W, d, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        Tensor dgama = new Tensor(1, 1, 1, C, true);
        Tensor dbeta = new Tensor(1, 1, 1, C, true);
        CUDAManager cudaManager = new CUDAManager(0);
        BNKernel2 kernel = new BNKernel2(BNType.conv_bn, C, H, W, cudaManager);
        for (int i = 0; i < 1; i++) {
            kernel.forward(RunModel.TRAIN, gama, beta, input, output);
            JCudaDriver.cuCtxSynchronize();
            kernel.backward(input, delta, diff, gama, dgama, dbeta);
        }
        output.syncHost();
        diff.syncHost();
        PrintUtils.printImage(input.getData());
        System.out.println("");
        System.out.println("=======output==============");
        PrintUtils.printImage(output.getData());
        System.out.println("");
        System.out.println("=======diff==============");
        PrintUtils.printImage(diff.getData());
        float[][][][] x_cpu = MatrixUtils.transform(x, N, C, H, W);
        float[][][][] d_cpu = MatrixUtils.transform(d, N, C, H, W);
        float[][][][] out_cpu = new float[N][C][H][W];
        float[][][][] diff_cpu = new float[N][C][H][W];
        float[] dgama_cpu = new float[C];
        float[] dbeta_cpu = new float[C];
        kernel.foward_cpu(x_cpu, out_cpu, d_cpu, diff_cpu, gama.getData(), beta.getData(), dgama_cpu, dbeta_cpu, 1);
        //    	System.out.println("");
        //    	PrintUtils.printImage(MatrixUtils.transform(out_cpu));
        //
        //    	System.out.println("==========diff===========");
        //
        //    	PrintUtils.printImage(diff_cpu);
        //
        //
        //    	System.out.println("==========gd===========");
        //
        //    	PrintUtils.printImage(dgama_cpu);
        //
        //
        //    	System.out.println("==========bd===========");
        //
        //    	PrintUtils.printImage(dbeta_cpu);
    }

    public static void test1d() {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 10;
        //    	float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        float[] x = new float[]{56.773f, -7.231f, 39.634f, 24.728f, -17.959f, 55.251f, -52.316f, -36.322f, -29.619f, 55.24f, 26.773f, -1.231f, 19.634f, 4.728f, 7.958f, -65.251f, 52.316f, -36.322f, -23.619f, -5.247f};
        float[] d = RandomUtils.val(N * C * H * W, 1f);
        float[] g = MatrixUtils.one(W);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor gama = new Tensor(1, 1, 1, W, g, true);
        Tensor beta = new Tensor(1, 1, 1, W, true);
        Tensor delta = new Tensor(N, C, H, W, d, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        Tensor dgama = new Tensor(1, 1, 1, W, true);
        Tensor dbeta = new Tensor(1, 1, 1, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        BNKernel2 kernel = new BNKernel2(BNType.fully_bn, C, H, W, cudaManager);
        for (int i = 0; i < 1; i++) {
            kernel.forward(RunModel.TRAIN, gama, beta, input, output);
            kernel.backward(input, delta, diff, gama, dgama, dbeta);
        }
        output.syncHost();
        diff.syncHost();
        PrintUtils.printImage(output.getData());
        System.out.println("");
        System.out.println("=======diff==============");
        PrintUtils.printImage(diff.getData());
        //    	float eta = 0.000001f;
        //
        //    	Tensor input1 = new Tensor(N, C, H, W, MatrixOperation.add(x, eta), true);
        //    	Tensor input2 = new Tensor(N, C, H, W, MatrixOperation.subtraction(x, eta), true);
        //    	Tensor output1 = new Tensor(N, C, H, W, true);
        //    	Tensor output2 = new Tensor(N, C, H, W, true);
        //
        //
        //    	System.out.println("gradientCheck:"+kernel.gradientCheck(input1, input2, dgama, dbeta, output1, output2, diff, eta));
        //    	System.out.println(JsonUtils.toJson(output.data));
        //    	System.out.println(JsonUtils.toJson(diff.data));
        float[][][][] x_cpu = MatrixUtils.transform(x, N, C, H, W);
        float[][][][] d_cpu = MatrixUtils.transform(d, N, C, H, W);
        float[][][][] out_cpu = new float[N][C][H][W];
        float[][][][] diff_cpu = new float[N][C][H][W];
        float[] dgama_cpu = new float[W];
        float[] dbeta_cpu = new float[W];
        kernel.foward_cpu(x_cpu, out_cpu, d_cpu, diff_cpu, gama.getData(), beta.getData(), dgama_cpu, dbeta_cpu, 0);
        PrintUtils.printImage(out_cpu);
        System.out.println("=====================");
        PrintUtils.printImage(diff_cpu);
    }

    public void initFunction() {
        try {
            if (mean_function == null) {
                mean_function = getCudaManager().getLocalFunctionByModule("MathKernel2.cu", "mean_full");
            }
            if (fast_mean_function == null) {
                fast_mean_function = getCudaManager().getLocalFunctionByModule("MathKernel2.cu", "fast_mean_kernel");
            }
            if (var_function == null) {
                var_function = getCudaManager().getLocalFunctionByModule("MathKernel2.cu", "var_full");
            }
            if (fast_var_function == null) {
                fast_var_function = getCudaManager().getLocalFunctionByModule("MathKernel2.cu", "fast_variance_kernel");
            }
            if (normalize_function == null) {
                normalize_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "normalize_kernel");
            }
            if (normalize_test_function == null) {
                normalize_test_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "normalize_test_kernel");
            }
            if (mwa_function == null) {
                mwa_function = getCudaManager().getLocalFunctionByModule("MathKernel2.cu", "mwa");
            }
            if (culOutput_function == null) {
                culOutput_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "culOutput_cov");
            }
            if (computeDelta_full_function == null) {
                computeDelta_full_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "computeDelta_full");
            }
            if (meanDzSum_function == null) {
                meanDzSum_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "meanDzSum");
            }
            if (computeDiff_function == null) {
                computeDiff_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "computeDiff");
            }
            /**
             * fast function

             */
            if (dgama_function == null) {
                dgama_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "dgama_kernel");
            }
            if (dbeta_function == null) {
                dbeta_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "dbeta_kernel");
            }
            if (dxhat_function == null) {
                dxhat_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "dxhat_kernel");
            }
            if (dx_function == null) {
                dx_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "dx_kernel");
            }
            if (dx_full_function == null) {
                dx_full_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "dx_kernel_full");
            }
            if (fast_mean_xhat_function == null) {
                fast_mean_xhat_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "fast_mean_xhat_kernel");
            }
            if (mean_xhat_function == null) {
                mean_xhat_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "mean_xhat_kernel");
            }
            if (fast_mean_dxhat_function == null) {
                fast_mean_dxhat_function = getCudaManager().getLocalFunctionByModule("BNKernel2.cu", "fast_mean_dxhat_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    private void initKernel() {
        /**
         * 申请向前传播参数显存

         */
        this.d_mean = CUDAMemoryManager.getDevice(meanNumber);
        this.d_var = CUDAMemoryManager.getDevice(meanNumber);
        this.d_std = CUDAMemoryManager.getDevice(meanNumber);
        this.d_runingMean = CUDAMemoryManager.getDevice(meanNumber);
        this.d_runingVar = CUDAMemoryManager.getDevice(meanNumber);
        this.d_mean_dz = CUDAMemoryManager.getDevice(meanNumber);
        this.d_mean_dzxz = CUDAMemoryManager.getDevice(meanNumber);
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

    public void initForward(RunModel RUN_MODEL, Tensor input, Tensor gama, Tensor beta, Tensor output) {
        if (input.getShape()[0] != this.N) {
            this.N = input.getShape()[0];
            if (bnType == BNType.fully_bn) {
                /**
                 * float* x,float* mean,int number,int width

                 */
                meanParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}));
                /**
                 * float* x,float* mean,float* var,int number,int width

                 */
                varParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_std), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}));
            } else {
                /**
                 * float *x, int batch, int filters, int spatial, float *mean

                 */
                fastMeanParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_mean));
                /**
                 * float *x, float *mean, int batch, int filters, int spatial, float *variance

                 */
                fastVarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_var), Pointer.to(d_std));
            }
            /**
             * float* mean,float* var,float* runingMean,float* runingStd,int n

             */
            mwaParameters = Pointer.to(Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_runingMean), Pointer.to(d_runingVar), Pointer.to(new int[]{meanNumber}), Pointer.to(new float[]{momentum}));
            int spatial = 1;
            if (bnType == BNType.conv_bn) {
                spatial = H * W;
            }
            d_z = CUDAMemoryManager.getDevice(N * C * H * W);
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalizeParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(d_z), Pointer.to(output.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(gama.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{meanNumber}), Pointer.to(new int[]{spatial}), Pointer.to(new float[]{eta}));
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalize_test_Parameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(d_z), Pointer.to(output.getGpuData()), Pointer.to(d_runingMean), Pointer.to(d_runingVar), Pointer.to(gama.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{meanNumber}), Pointer.to(new int[]{spatial}), Pointer.to(new float[]{eta}));
        }
    }

    public void initBackward(Tensor input, Tensor delta, Tensor diff, Tensor gama, Tensor dgama, Tensor dbeta) {
        if (dgamaParameters == null) {
            if (bnType == BNType.fully_bn) {
                /**
                 * 设置入参
                 * float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int width

                 */
                computeDelta_full_Parameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dgama.getGpuData()), Pointer.to(dbeta.getGpuData()), Pointer.to(d_z), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}));
                /**
                 * 设置入参
                 * float *z, float *std, float *diff, float *mean_dz, float *mean_dzxz, int batch, int filters

                 */
                dx_fullParameters = Pointer.to(Pointer.to(d_z), Pointer.to(d_std), Pointer.to(diff.getGpuData()), Pointer.to(d_mean_dz), Pointer.to(d_mean_dzxz), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}));
            } else {
                /**
                 * 设置入参
                 * int N, float *z, float *std,float *diff, float *mean_dz, float *mean_dzxz, int batch, int filters, int spatial

                 */
                dxParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(d_z), Pointer.to(d_std), Pointer.to(diff.getGpuData()), Pointer.to(d_mean_dz), Pointer.to(d_mean_dzxz), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
                /**
                 * 设置入参
                 * float *x_norm, float *delta, int batch, int c, int size, float *dgama

                 */
                dgamaParameters = Pointer.to(Pointer.to(d_z), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(dgama.getGpuData()));
                /**
                 * 设置入参
                 * float *dbeta, float *delta, int batch, int c, int size

                 */
                dbetaParameters = Pointer.to(Pointer.to(dbeta.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
            }
            int spatial = 1;
            if (bnType == BNType.conv_bn) {
                spatial = H * W;
            }
            /**
             * float *z,float *dz, int batch, int filters, int spatial, float *mean_dz,float *mean_dzxz

             */
            fast_mean_xhat_Parameters = Pointer.to(Pointer.to(d_z), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(d_mean_dz));
            /**
             * float *z,float *dz, int batch, int filters, int spatial,float *mean_dzxz

             */
            fast_mean_dxhat_Parameters = Pointer.to(Pointer.to(d_z), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{spatial}), Pointer.to(d_mean_dzxz));
            /**
             * float *z,float *dz,float *mean_dz,float *mean_dzxz,int number,int channel,int height,int width

             */
            mean_xhat_Parameters = Pointer.to(Pointer.to(d_z), Pointer.to(diff.getGpuData()), Pointer.to(d_mean_dz), Pointer.to(d_mean_dzxz), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
            /**
             * 设置入参
             * int N, float *delta, float *dz, float *gama, int filters, int spatial

             */
            dxhatParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(gama.getGpuData()), Pointer.to(new int[]{meanNumber}), Pointer.to(new int[]{spatial}));
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(RunModel RUN_MODEL, Tensor gama, Tensor beta, Tensor input, Tensor output) {
        //		long start = System.nanoTime();
        initForward(RUN_MODEL, input, gama, beta, output);
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
            normalize_train(input, gama, beta, output);
        } else {
            normalize_test(input, gama, beta, output);
        }
    }

    public void mean() {
        try {
            cuLaunchKernel(mean_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    meanParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void var() {
        try {
            cuLaunchKernel(var_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    varParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fast_mean() {
        try {
            cuLaunchKernel(fast_mean_function, meanNumber, 1, 1,      // Grid dimension
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
            cuLaunchKernel(fast_var_function, meanNumber, 1, 1,      // Grid dimension
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
            cuLaunchKernel(mwa_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
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

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gama, Tensor dgama, Tensor dbeta) {
        initBackward(input, delta, diff, gama, dgama, dbeta);
        if (bnType == BNType.fully_bn) {
            computeDelta_full();
        } else {
            computeDgama();
            computeDbeta();
        }
        computeDxhat();
        //		computeMeanXhat();
        //
        //		computeMeanDXhat();
        computeMeanXhat2();
        //
        //		System.out.println("d_mean_dz");
        //		float[] tmp = new float[C];
        //		showDM(d_mean_dz, tmp);
        //
        //		System.out.println("d_mean_dzxz");
        //		float[] tmp2 = new float[C];
        //		showDM(d_mean_dzxz, tmp2);
        //
        if (bnType == BNType.fully_bn) {
            computeDx_full();
        } else {
            computeDx();
        }
        //		diff.showDM();
    }

    private void computeDgama() {
        cuLaunchKernel(dgama_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dgamaParameters, null // Kernel- and extra parameters
        );
    }

    private void computeDbeta() {
        cuLaunchKernel(dbeta_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dbetaParameters, null // Kernel- and extra parameters
        );
    }

    private void computeDelta_full() {
        cuLaunchKernel(computeDelta_full_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                computeDelta_full_Parameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeDxhat() {
        cuLaunchKernel(dxhat_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dxhatParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeMeanXhat() {
        cuLaunchKernel(fast_mean_xhat_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fast_mean_xhat_Parameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeMeanDXhat() {
        cuLaunchKernel(fast_mean_dxhat_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fast_mean_dxhat_Parameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeMeanXhat2() {
        //		System.out.println("z");
        //		float[] tmp1 = new float[N * C * H * W];
        //		showDM(d_z, tmp1);
        cuLaunchKernel(mean_xhat_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                mean_xhat_Parameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeDx() {
        cuLaunchKernel(dx_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dxParameters, null // Kernel- and extra parameters
        );
    }

    private void computeDx_full() {
        cuLaunchKernel(dx_full_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                dx_fullParameters, null // Kernel- and extra parameters
        );
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

    public void foward_cpu(float[][][][] x, float[][][][] out, float[][][][] delta, float[][][][] diff, float[] gama, float[] beta, float[] dgama, float[] dbeta, int type) {
        float[] mean = new float[C];
        float[] var = new float[C];
        MatrixOperation.meanV2(x, mean, type);
        /**
         * 计算标准差
         * var = 1/m ∑(x - mean)^2

         */
        MatrixOperation.varV2(x, mean, var, type);
        /**
         * std = (var + eta)^1/2
         * zi = (xi - mean) / (std)

         */
        float[][][][] z = this.culOutput(x, out, mean, var, gama, beta, type);
        //		float[] dvar = new float[C];
        //
        //		float[] dmu = new float[C];
        //
        //		computeDelta_cpu(delta, z, gama, dgama, dbeta, diff, type);
        /**
         * 原论文公式
         * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
         * 使用darknet公式
         * dmean = (∑ dxhat * -1 / (var + eta)^1/2)

         */
        //		meanDzSum_cpu(dvar, mean, var, dmu, x, diff, type);
        //		System.out.println("dgama:"+JsonUtils.toJson(dgama));
        //		System.out.println("dbeta:"+JsonUtils.toJson(dbeta));
        //		System.out.println("dgama error:"+CheckArrayUtils.oneCheck(dgama, this.dgama));
        //		System.out.println("dbeta error:"+CheckArrayUtils.oneCheck(dbeta, this.dbeta));
        //		System.out.println("dgama-cpu:"+JsonUtils.toJson(dgama));
        //		System.out.println("dbeta-cpu:"+JsonUtils.toJson(dbeta));
        //		float[] z_cpu = MatrixUtils.transform(z);
        //		float[] z_gpu = new float[z_cpu.length];
        //		JCudaDriver.cuMemcpyDtoH(Pointer.to(z_gpu), d_z, z_gpu.length * Sizeof.FLOAT);
        //
        //		System.out.println("z error:"+CheckArrayUtils.oneCheck(z_cpu, z_gpu));
        //float[][][][] diff,float[][][][] x,float[] mean,float[] dmu,float[] std,float[] dvar
        //		System.out.println("dmu:");
        //		PrintUtils.printImage(dmu);
        //		System.out.println("");
        //		System.out.println("dvar:");
        //		PrintUtils.printImage(dvar);
        //		System.out.println("");
        //		computeDiff_cpu(diff, x, mean, var, dmu, dvar, type);
        ////		float[][][][] delta,float[][][][] z,float[][][][] diff,float[] std,float[] gama,float[] dgama,float[] dbeta
        ////		backward_caffe(delta, z, diff, std, gama, dgama, dbeta);
        //
        //		float[][][][] dx = new float[N][C][H][W];
        //
        //		float[] std = new float[C];
        //
        //		std(x, mean, std);
        //
        //		float[] std2 = new float[C];
        //		for(int ic = 0;ic<C;ic++) {
        //			std2[ic] = (float) Math.sqrt(var[ic] + eta);
        //		}
        //
        //		for(int ic = 0;ic<C;ic++) {
        //			System.out.println(std[ic] + ":" + std2[ic]);
        //		}
        //
        float[][][][] dxhat = new float[N][C][H][W];
        dxhat(dxhat, gama, delta);
        //
        //		dx2(dx, std, z, dxhat);
        //
        //		System.out.println("dx2:");
        //
        //		PrintUtils.printImage(dx);
        //
        //		System.out.println("");
        //
        float[] var2 = new float[C];
        var(x, mean, var2);
        float[] std3 = new float[C];
        for (int ic = 0; ic < C; ic++) {
            std3[ic] = (float) Math.sqrt(var2[ic] + eta);
        }
        System.out.println("dx3:");
        float[][][][] dx3 = new float[N][C][H][W];
        dx3(dx3, std3, z, dxhat);
        PrintUtils.printImage(MatrixUtils.transform(dx3));
    }

    private void computeDelta_cpu(float[][][][] delta, float[][][][] z, float[] gama, float[] dgama, float[] dbeta, float[][][][] diff, int type) {
        if (type == 1) {
            //    		System.out.println(C+":"+N + ":" + H + ":" + W);
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
        } else {
            for (int w = 0; w < W; w++) {
                dgama[w] = 0;
                dbeta[w] = 0;
                for (int m = 0; m < N; m++) {
                    // deltaGama = ∑ deta * z
                    dgama[w] += delta[m][0][0][w] * z[m][0][0][w];
                    // deltaBeta = ∑ deta
                    dbeta[w] += delta[m][0][0][w];
                    // dxhat = deta * gama
                    diff[m][0][0][w] = delta[m][0][0][w] * gama[w];
                }
            }
        }
    }

    /**
     * 原论文公式
     * <p>
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
     * <p>
     * 使用darknet公式
     * <p>
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
     */
    private void meanDzSum_cpu(float[] dvar, float[] mean, float[] var, float[] dmu, float[][][][] x, float[][][][] dz, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                float dvar_val = 0.0f;
                float dmu_val = 0.0f;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dvar_val += (x[n][c][h][w] - mean[c]) * dz[n][c][h][w];
                            dmu_val += -1.0f * dz[n][c][h][w] / (float) Math.sqrt(var[c] + eta);
                        }
                    }
                }
                dvar[c] = (float) (dvar_val * -0.5f * Math.pow(var[c] + eta, -1.5));
                dmu[c] = dmu_val;
            }
        } else {
            for (int w = 0; w < W; w++) {
                float dvar_val = 0.0f;
                float dmu_val = 0.0f;
                for (int n = 0; n < N; n++) {
                    dvar_val += (x[n][0][0][w] - mean[w]) * dz[n][0][0][w];
                    dmu_val += -1.0f * dz[n][0][0][w] / (float) Math.sqrt(var[w] + eta);
                }
                dvar[w] = (float) (dvar_val * -0.5f * Math.pow(var[w] + eta, -1.5));
                dmu[w] = dmu_val;
            }
        }
        //    	System.out.println("var_cpu:"+JsonUtils.toJson(var));
        //    	System.out.println("dmean_cpu:"+JsonUtils.toJson(dmu));
    }

    private void computeDiff_cpu(float[][][][] diff, float[][][][] x, float[] mean, float[] var, float[] dmu, float[] dvar, int type) {
        if (type == 1) {
            float scale = 1.0f / (N * H * W);
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
        } else {
            float scale = 1.0f / N;
            for (int m = 0; m < N; m++) {
                for (int w = 0; w < W; w++) {
                    // dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
                    diff[m][0][0][w] = diff[m][0][0][w] / (float) Math.sqrt(var[w] + eta) + 2.0f * dvar[w] * (x[m][0][0][w] - mean[w]) * scale + dmu[w] * scale;
                }
            }
        }
    }

    private float[][][][] culOutput(float[][][][] x, float[][][][] out, float[] m, float[] var, float[] gama, float[] beta, int type) {
        int N = x.length;
        int C = x[0].length;
        int H = x[0][0].length;
        int W = x[0][0][0].length;
        System.out.println(N + ":" + C + ":" + H + ":" + W);
        float[][][][] z = new float[N][C][H][W];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        if (type == 0) {
                            z[n][c][h][w] = (x[n][c][h][w] - m[w]) / (float) Math.sqrt(var[w] + eta);
                            out[n][c][h][w] = z[n][c][h][w] * gama[w] + beta[w];
                        } else {
                            z[n][c][h][w] = (x[n][c][h][w] - m[c]) / (float) Math.sqrt(var[c] + eta);
                            out[n][c][h][w] = z[n][c][h][w] * gama[c] + beta[c];
                        }
                    }
                }
            }
        }
        return z;
    }

    /**
     * @param x
     * @return
     * @Title: gradientCheck
     * @Description: TODO(这里用一句话描述这个方法的作用)
     * <p>
     * gradientCheck:
     * <p>
     * (f(x + eta) - f(x - eta)) / (2 * eta) ≈ f'(x)
     */
    public float gradientCheck(Tensor input1, Tensor input2, Tensor gama, Tensor beta, Tensor output1, Tensor output2, Tensor diff, float eta) {
        this.forward(RunModel.TRAIN, gama, beta, input1, output1);
        this.forward(RunModel.TRAIN, gama, beta, input2, output2);
        float[] data = MatrixOperation.division(MatrixOperation.subtraction(output1.syncHost(), output2.syncHost()), 2.0f * eta);
        return CheckArrayUtils.check(diff.syncHost(), data);
    }

    public void var(float[][][][] x, float[] mean, float[] var) {
        int sn = N * H * W;
        for (int c = 0; c < C; c++) {
            var[c] = 0;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        var[c] += (x[n][c][h][w] - mean[c]) * (x[n][c][h][w] - mean[c]);
                    }
                }
            }
            var[c] = var[c] / sn;
        }
    }

    public void std(float[][][][] x, float[] mean, float[] std) {
        int sn = N * H * W;
        for (int c = 0; c < C; c++) {
            std[c] = 0;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        std[c] += (x[n][c][h][w] - mean[c]) * (x[n][c][h][w] - mean[c]);
                    }
                }
            }
            std[c] = (float) Math.sqrt((std[c] + eta) / sn);
        }
    }

    public void dxhat(float[][][][] dxhat, float[] gama, float[][][][] delta) {
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dxhat[n][c][h][w] = delta[n][c][h][w] * gama[c];
                    }
                }
            }
        }
    }

    /**
     * std = sqrt(1/n∑(xi-mean)^2)
     * <p>
     * xhat = (xi - mean) / std
     * <p>
     * dxhat = gama * delta
     * <p>
     * dx = 1/(n*std) * (n * dxhat - ∑dxhat - xhat * ∑dxhatj*xhatj)
     */
    public void dx2(float[][][][] dx, float[] std, float[][][][] xhat, float[][][][] dxhat) {
        float[] dxhatSum = new float[C];
        float[] dxhat_Sum = new float[C];
        int sn = N * H * W;
        /**
         * ∑dxhatj*xhatj
         * ∑dxhat

         */
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dxhat_Sum[c] += dxhat[n][c][h][w] * xhat[n][c][h][w];
                        dxhatSum[c] += dxhat[n][c][h][w];
                    }
                }
            }
        }
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dx[n][c][h][w] = (sn * dxhat[n][c][h][w] - dxhatSum[c] - xhat[n][c][h][w] * dxhat_Sum[c]) / (sn * std[c]);
                    }
                }
            }
        }
    }

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
    public void dx3(float[][][][] dx, float[] std, float[][][][] xhat, float[][][][] dxhat) {
        float[] dxhatSum = new float[C];
        float[] dxhat_Sum = new float[C];
        float[] tmp1 = new float[N * C * H * W];
        showDM(d_z, tmp1);
        int sn = N * H * W;
        float[][][][] t = MatrixUtils.transform(tmp1, N, C, H, W);
        /**
         * ∑dxhatj*xhatj
         * ∑dxhat

         */
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dxhatSum[c] += dxhat[n][c][h][w];
                        dxhat_Sum[c] += dxhat[n][c][h][w] * t[n][c][h][w];
                    }
                }
            }
        }
        System.out.println(JsonUtils.toJson(MatrixUtils.transform(xhat)));
        System.out.println(CheckArrayUtils.check(tmp1, MatrixUtils.transform(xhat)));
        for (int c = 0; c < C; c++) {
            float m1 = dxhatSum[c] / sn;
            float m2 = dxhat_Sum[c] / sn;
            //			System.out.println(m2);
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dx[n][c][h][w] = 1 / std[c] * (dxhat[n][c][h][w] - m1 - xhat[n][c][h][w] * m2);
                    }
                }
            }
        }
    }
}

