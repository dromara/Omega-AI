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

public class BNKernel extends BNBaseKernel {
    private BNType bnType;
    private int C;
    private int H;
    private int W;
    private int meanNumber = 0;
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
    private CUfunction full_dmean_ov_function;
    private CUfunction full_dvar_function;
    private CUfunction fast_dmean_function;
    private CUfunction fast_dmean_ov_function;
    private CUfunction fast_dvar_function;
    private CUfunction dx_function;
    private CUfunction dx_full_function;
    private CUfunction computeDParams_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private float eta = 1e-5f;
    private float momentum = 0.9f;
    /**
     * 前向参数
     */
    private CUdeviceptr d_z;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_var;
    private CUdeviceptr d_runingMean;
    private CUdeviceptr d_runingVar;
    /**
     * 反向参数
     */
    private CUdeviceptr d_dmean;
    private CUdeviceptr d_dvar;
    /**
     * 前向方法
     */
    private Pointer meanParameters;
    private Pointer varParameters;
    private Pointer fastMeanParameters;
    private Pointer fastVarParameters;
    private Pointer normalizeParameters;
    private Pointer normalize_test_Parameters;
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
    private Pointer computeDParams_Parameters;
    private Pointer dxhatParameters;
    private Pointer fullDmeanParameters;
    private Pointer fullDMeanOVParameters;
    private Pointer fullDvarParameters;
    private Pointer fastDmeanParameters;
    private Pointer fastDvarParameters;
    private Pointer fastDMeanOVParameters;
    private Pointer dxParameters;
    private Pointer dx_fullParameters;

    public BNKernel(BNType bnType, int C, int H, int W, CUDAManager cudaManager) {
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
        //    	test1d();
        test2d();
        //    	System.out.println(gradientCheck());
    }

    public static void test2d() {
        int N = 2;
        int C = 3;
        int H = 5;
        int W = 5;
        //    	float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        float[] x = new float[]{0.9827f, 0.5268f, 0.4057f, 0.2853f, 0.1708f, 0.4791f, 0.5626f, 0.1290f, 0.9540f, 0.7471f, 0.5806f, 0.8789f, 0.9766f, 0.8142f, 0.9557f, 0.2814f, 0.7667f, 0.5963f, 0.0016f, 0.5944f, 0.4617f, 0.0975f, 0.3558f, 0.3318f, 0.5196f, 0.7558f, 0.7438f, 0.4061f, 0.2737f, 0.1826f, 0.7600f, 0.3608f, 0.3924f, 0.2537f, 0.7536f, 0.7980f, 0.5246f, 0.6428f, 0.0571f, 0.9973f, 0.7106f, 0.5854f, 0.3122f, 0.2741f, 0.2868f, 0.4628f, 0.2696f, 0.0436f, 0.1222f, 0.4933f, 0.5372f, 0.4992f, 0.2837f, 0.8462f, 0.2095f, 0.1916f, 0.1830f, 0.1934f, 0.8305f, 0.0776f, 0.9014f, 0.1835f, 0.7673f, 0.0999f, 0.5783f, 0.7816f, 0.2961f, 0.9230f, 0.3454f, 0.6030f, 0.4821f, 0.0113f, 0.9629f, 0.8698f, 0.8440f, 0.9763f, 0.7661f, 0.2085f, 0.4248f, 0.7407f, 0.5092f, 0.5272f, 0.8521f, 0.1649f, 0.9759f, 0.9084f, 0.3206f, 0.3061f, 0.9648f, 0.3377f, 0.6753f, 0.6662f, 0.4570f, 0.9556f, 0.0918f, 0.8788f, 0.6432f, 0.4928f, 0.8778f, 0.5665f, 0.7979f, 0.5639f, 0.5970f, 0.4987f, 0.1227f, 0.4963f, 0.6865f, 0.5728f, 0.1927f, 0.1199f, 0.5015f, 0.0221f, 0.0826f, 0.0077f, 0.0568f, 0.7569f, 0.7684f, 0.1536f, 0.4406f, 0.2919f, 0.3006f, 0.9501f, 0.1994f, 0.3314f, 0.5612f, 0.3303f, 0.8773f, 0.3262f, 0.1926f, 0.8667f, 0.3360f, 0.5357f, 0.3332f, 0.2044f, 0.5538f, 0.0607f, 0.2203f, 0.7994f, 0.6357f, 0.6469f, 0.8163f, 0.7764f, 0.6821f, 0.6798f, 0.0553f, 0.0609f, 0.2305f, 0.7183f, 0.8135f, 0.7688f};
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
        BNKernel kernel = new BNKernel(BNType.conv_bn, C, H, W, cudaManager);
        for (int i = 0; i < 1; i++) {
            kernel.forward(RunModel.TRAIN, gama, beta, input, output);
            JCudaDriver.cuCtxSynchronize();
            kernel.backward(input, delta, diff, gama, dgama, dbeta);
        }
        output.syncHost();
        diff.syncHost();
        PrintUtils.printImage(input.getData());
        System.out.println("");
        float[][][][] x_cpu = MatrixUtils.transform(x, N, C, H, W);
        float[][][][] d_cpu = MatrixUtils.transform(d, N, C, H, W);
        float[][][][] out_cpu = new float[N][C][H][W];
        float[][][][] diff_cpu = new float[N][C][H][W];
        float[] dgama_cpu = new float[C];
        float[] dbeta_cpu = new float[C];
        kernel.foward_cpu(x_cpu, out_cpu, d_cpu, diff_cpu, gama.getData(), beta.getData(), dgama_cpu, dbeta_cpu, 1);
        System.out.println("=======output==============");
        PrintUtils.printImage(output.getData());
        System.out.println("");
        System.out.println("=======output-cpu==============");
        System.out.println(JsonUtils.toJson(MatrixUtils.transform(out_cpu)));
        System.out.println("==========diff-cpu===========");
        System.out.println(JsonUtils.toJson(MatrixUtils.transform(diff_cpu)));
        System.out.println("=======diff==============");
        PrintUtils.printImage(diff.getData());
        System.out.println("==========gd===========");
        PrintUtils.printImage(dgama_cpu);
        System.out.println("");
        System.out.println("output-error:" + CheckArrayUtils.check(output.getData(), MatrixUtils.transform(out_cpu)));
        System.out.println("diff-error:" + CheckArrayUtils.check(diff.getData(), MatrixUtils.transform(diff_cpu)));
    }

    public static void test1d() {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 10;
        //float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.01f);
        //float[] x = new float[] {1f,1f,1f,2f,-3f,5f,0.1f,-0.3f,0.5f,1.2f,-1.3f,1.5f};
        float[] x = new float[]{56.773f, -7.231f, 39.634f, 24.728f, -17.959f, 55.251f, -52.316f, -36.322f, -29.619f, 55.24f, 26.773f, -1.231f, 19.634f, 4.728f, 7.958f, -65.251f, 52.316f, -36.322f, -23.619f, -5.247f};
        float[] d = RandomUtils.val(N * C * H * W, 1.0f);
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
        BNKernel kernel = new BNKernel(BNType.fully_bn, C, H, W, cudaManager);
        for (int i = 0; i < 1; i++) {
            kernel.forward(RunModel.TRAIN, gama, beta, input, output);
            kernel.backward(input, delta, diff, gama, dgama, dbeta);
        }
        output.syncHost();
        diff.syncHost();
        dgama.syncHost();
        float[][][][] x_cpu = MatrixUtils.transform(x, N, C, H, W);
        float[][][][] d_cpu = MatrixUtils.transform(d, N, C, H, W);
        float[][][][] out_cpu = new float[N][C][H][W];
        float[][][][] diff_cpu = new float[N][C][H][W];
        float[] dgama_cpu = new float[W];
        float[] dbeta_cpu = new float[W];
        kernel.foward_cpu(x_cpu, out_cpu, d_cpu, diff_cpu, gama.getData(), beta.getData(), dgama_cpu, dbeta_cpu, 0);
        System.out.println("output-gpu:");
        System.out.println(JsonUtils.toJson(output.getData()));
        System.out.println("output-cpu:");
        System.out.println(JsonUtils.toJson(MatrixUtils.transform(out_cpu)));
        System.out.println("=======diff-gpu==============");
        System.out.println(JsonUtils.toJson(diff.getData()));
        System.out.println("=======diff-cpu==============");
        System.out.println(JsonUtils.toJson(MatrixUtils.transform(diff_cpu)));
        System.out.println(JsonUtils.toJson(dgama.getData()));
        System.out.println("output-error:" + CheckArrayUtils.check(output.getData(), MatrixUtils.transform(out_cpu)));
        System.out.println("diff-error:" + CheckArrayUtils.check(diff.getData(), MatrixUtils.transform(diff_cpu)));
    }

    public static float gradientCheck() {
        int N = 4;
        int C = 1;
        int H = 1;
        int W = 3;
        float eta = 1e-6f;
        //float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        //float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.01f);
        float[] x = new float[]{1f, 1f, 1f, 2f, -3f, 5f, 0.1f, -0.3f, 0.5f, 1.2f, -1.3f, 1.5f};
        float[] x1 = MatrixOperation.add(x, eta);
        float[] x2 = MatrixOperation.subtraction(x, eta);
        float[] d = RandomUtils.val(N * C * H * W, 1f);
        float[] g = MatrixUtils.one(W);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor input1 = new Tensor(N, C, H, W, x1, true);
        Tensor input2 = new Tensor(N, C, H, W, x2, true);
        Tensor gama = new Tensor(1, 1, 1, W, g, true);
        Tensor beta = new Tensor(1, 1, 1, W, true);
        Tensor delta = new Tensor(N, C, H, W, d, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor output1 = new Tensor(N, C, H, W, true);
        Tensor output2 = new Tensor(N, C, H, W, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        Tensor dgama = new Tensor(1, 1, 1, W, true);
        Tensor dbeta = new Tensor(1, 1, 1, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        BNKernel kernel = new BNKernel(BNType.fully_bn, C, H, W, cudaManager);
        BNKernel kernel1 = new BNKernel(BNType.fully_bn, C, H, W, cudaManager);
        kernel.forward(RunModel.TRAIN, gama, beta, input1, output1);
        kernel1.forward(RunModel.TRAIN, gama, beta, input2, output2);
        output1.syncHost();
        output2.syncHost();
        float[] diff_check = MatrixOperation.division(MatrixOperation.subtraction(output1.getData(), output2.getData()), 2 * eta);
        System.out.println(JsonUtils.toJson(diff_check));
        BNKernel kernel2 = new BNKernel(BNType.fully_bn, C, H, W, cudaManager);
        kernel2.forward(RunModel.TRAIN, gama, beta, input, output);
        kernel2.backward(input, delta, diff, gama, dgama, dbeta);
        diff.syncHost();
        System.out.println(JsonUtils.toJson(diff.getData()));
        return CheckArrayUtils.check(diff.getData(), diff_check);
    }

    public void initFunction() {
        try {
            if (computeDParams_function == null) {
                computeDParams_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "computeDParams");
            }
            if (mean_function == null) {
                mean_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "mean_full");
            }
            if (fast_mean_function == null) {
                fast_mean_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "fast_mean_kernel");
            }
            if (var_function == null) {
                var_function = getCudaManager().getLocalFunctionByModule("MathKernel.cu", "var_full");
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
            if (full_dmean_ov_function == null) {
                full_dmean_ov_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "full_mean_delta_ov_kernel");
            }
            if (fast_dmean_function == null) {
                fast_dmean_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "fast_mean_delta_kernel");
            }
            if (fast_dmean_ov_function == null) {
                fast_dmean_ov_function = getCudaManager().getLocalFunctionByModule("BNKernel.cu", "fast_mean_delta_ov_kernel");
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

    private void initKernel() {
        /**
         * 申请向前传播参数显存

         */
        this.d_mean = CUDAMemoryManager.getDevice(meanNumber);
        this.d_var = CUDAMemoryManager.getDevice(meanNumber);
        this.d_runingMean = CUDAMemoryManager.getDevice(meanNumber);
        this.d_runingVar = CUDAMemoryManager.getDevice(meanNumber);
        /**
         * 申请反向传播参数显存

         */
        this.d_dmean = CUDAMemoryManager.getDevice(meanNumber);
        this.d_dvar = CUDAMemoryManager.getDevice(meanNumber);
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
                varParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}));
            } else {
                /**
                 * float *x, int batch, int filters, int spatial, float *mean

                 */
                fastMeanParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_mean));
                /**
                 * float *x, float *mean, int batch, int filters, int spatial, float *variance

                 */
                fastVarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_var));
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
                 * float *x_norm, float *delta, int batch, int c, int size, float *dgama

                 */
                dgamaParameters = Pointer.to(Pointer.to(d_z), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(dgama.getGpuData()));
                /**
                 * 设置入参
                 * float *dbeta, float *delta, int batch, int c, int size

                 */
                dbetaParameters = Pointer.to(Pointer.to(dbeta.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}));
                /**
                 * 设置入参
                 * float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int width

                 */
                computeDelta_full_Parameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dgama.getGpuData()), Pointer.to(dbeta.getGpuData()), Pointer.to(d_z), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}));
                /**
                 * 设置入参
                 * float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, float *diff

                 */
                dx_fullParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_dmean), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}), Pointer.to(diff.getGpuData()));
                /**
                 * 设置入参
                 * float *dxhat, float *variance, int batch, int filters, float *mean_delta

                 */
                fullDmeanParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}), Pointer.to(d_dmean));
                /**
                 * 设置入参
                 * float *x, float *dxhat, float *mean, float *variance, int batch, int filters, float *variance_delta

                 */
                fullDvarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}), Pointer.to(d_dvar));
                /**
                 * 设置入参
                 * float *dxhat, float *variance, float *mean, float *x, float *dvar, int batch, int filters, float *mean_delta

                 */
                fullDMeanOVParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(d_var), Pointer.to(d_mean), Pointer.to(input.getGpuData()), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{W}), Pointer.to(d_dmean));
            } else {
                /**
                 * 设置入参
                 * int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *diff

                 */
                dxParameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_dmean), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(diff.getGpuData()));
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
                /**
                 * 设置入参
                 * float *dxhat, float *variance, int batch, int filters, int spatial, float *mean_delta

                 */
                fastDmeanParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dmean));
                /**
                 * 设置入参
                 * float *dxhat, float *variance, float *mean, float *x, float *dvar, int batch, int filters, int spatial, float *mean_delta

                 */
                fastDMeanOVParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(d_var), Pointer.to(d_mean), Pointer.to(input.getGpuData()), Pointer.to(d_dvar), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dmean));
                /**
                 * 设置入参
                 * float *x, float *dxhat, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta

                 */
                fastDvarParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(d_dvar));
            }
            int spatial = 1;
            if (bnType == BNType.conv_bn) {
                spatial = H * W;
            }
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
            normalize(input, gama, beta, output);
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

    public void normalize(Tensor input, Tensor gama, Tensor beta, Tensor output) {
        try {
            cuLaunchKernel(normalize_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
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
        //		long start = System.nanoTime();
        initBackward(input, delta, diff, gama, dgama, dbeta);
        if (bnType == BNType.fully_bn) {
            //			computeDelta_full();
            computeDgama();
            computeDbeta();
        } else {
            computeDgama();
            computeDbeta();
            //			computeDParams(delta, dgama, dbeta);
        }
        //		System.out.println((System.nanoTime() - start) / 1e6 + "ms.1");
        //		long start2 = System.nanoTime();
        computeDxhat();
        //
        //		System.out.println("in--->");
        //
        //		diff.showDM();
        //
        //		dgama.showDM();
        //
        //		dbeta.showDM();
        if (bnType == BNType.fully_bn) {
            computeFullDvar();
            computeFullDmean();
            //			computeFullDmeanOV();
        } else {
            computeDvar();
            computeDmean();
            //			computeDmeanOV();
        }
        //		float[] dmeanData = new float[C];
        //
        //		showDM(d_dmean, dmeanData);
        //
        //		System.out.println("dvar-gpu:");
        //		float[] dvarData = new float[meanNumber];
        //		showDM(d_dvar, dvarData);
        //
        //		System.out.println("dmean-gpu:");
        //		float[] dmeanData = new float[meanNumber];
        //		showDM(d_dmean, dmeanData);
        //		System.out.println((System.nanoTime() - start2) / 1e6 + "ms.2");
        //		long start3 = System.nanoTime();
        if (bnType == BNType.fully_bn) {
            computeDx_full();
        } else {
            computeDx();
        }
        //		diff.showDM();
        //		System.out.println((System.nanoTime() - start3) / 1e6 + "ms.3");
        //
        //		System.out.println("===========>"+id);
        //		System.out.println((System.nanoTime() - start)/1e6+"ms.backward");
    }

    private void computeDParams(Tensor delta, Tensor dgama, Tensor dbeta) {
        /**
         * float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int channel,int height,int width

         */
        computeDParams_Parameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dgama.getGpuData()), Pointer.to(dbeta.getGpuData()), Pointer.to(d_z), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
        cuLaunchKernel(computeDParams_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                computeDParams_Parameters, null // Kernel- and extra parameters
        );
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

    private void computeDmean() {
        cuLaunchKernel(fast_dmean_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fastDmeanParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeDmeanOV() {
        cuLaunchKernel(fast_dmean_ov_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fastDMeanOVParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeFullDmean() {
        cuLaunchKernel(full_dmean_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fullDmeanParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeFullDmeanOV() {
        cuLaunchKernel(full_dmean_ov_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fullDMeanOVParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeDvar() {
        cuLaunchKernel(fast_dvar_function, meanNumber, 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fastDvarParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    private void computeFullDvar() {
        cuLaunchKernel(full_dvar_function, this.CAFFE_GET_BLOCKS(meanNumber), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                fullDvarParameters, null // Kernel- and extra parameters
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

    public void foward_cpu(float[][][][] x, float[][][][] out, float[][][][] delta, float[][][][] diff, float[] gamma, float[] beta, float[] dgamma, float[] dbeta, int type) {
        int S = W;
        if (type == 1) {
            S = C;
        }
        float[] mean = new float[S];
        float[] var = new float[S];
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
        float[][][][] xhat = this.culOutput(x, out, mean, var, gamma, beta, type);
        backward_cpu(x, mean, var, xhat, delta, gamma, diff, dgamma, dbeta, type);
        //		computeDelta_cpu(delta, z, gama, dgama, dbeta, diff, type);
        //
        //		/**
        //		 * 原论文公式
        //		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
        //		 * 使用darknet公式
        //		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
        //		 */
        //		meanDzSum_cpu(dvar, mean, var, dmu, x, diff, type);
        //
        ////		System.out.println("dgama:"+JsonUtils.toJson(dgama));
        ////		System.out.println("dbeta:"+JsonUtils.toJson(dbeta));
        //
        ////		System.out.println("dgama error:"+CheckArrayUtils.oneCheck(dgama, this.dgama));
        ////		System.out.println("dbeta error:"+CheckArrayUtils.oneCheck(dbeta, this.dbeta));
        ////		System.out.println("dgama-cpu:"+JsonUtils.toJson(dgama));
        ////		System.out.println("dbeta-cpu:"+JsonUtils.toJson(dbeta));
        //
        ////		float[] z_cpu = MatrixUtils.transform(z);
        ////		float[] z_gpu = new float[z_cpu.length];
        //
        ////		JCudaDriver.cuMemcpyDtoH(Pointer.to(z_gpu), d_z, z_gpu.length * Sizeof.FLOAT);
        ////
        ////		System.out.println("z error:"+CheckArrayUtils.oneCheck(z_cpu, z_gpu));
        //
        //		//float[][][][] diff,float[][][][] x,float[] mean,float[] dmu,float[] std,float[] dvar
        //		System.out.println("dmu:");
        //		PrintUtils.printImage(dmu);
        //		System.out.println("");
        //		System.out.println("dvar:");
        //		PrintUtils.printImage(dvar);
        //		System.out.println("");
        //		computeDiff_cpu(diff, x, mean, var, dmu, dvar, type);
        //		System.out.println(JsonUtils.toJson(MatrixUtils.transform(diff)));
        //		float[][][][] delta,float[][][][] z,float[][][][] diff,float[] std,float[] gama,float[] dgama,float[] dbeta
        //		backward_caffe(delta, z, diff, std, gama, dgama, dbeta);
        //
        //		float[][][][] dx = new float[N][C][H][W];
        //
        //		float[] std = new float[C];
        //
        //		std(x, mean, std);
        //
        //		float[][][][] dxhat = new float[N][C][H][W];
        //
        //		dxhat(dxhat, gama, delta);
        //
        //		dx2(dx, std, z, dxhat);
        //
        //		System.out.println("dx2:");
        //
        //		PrintUtils.printImage(diff);
        //
    }

    /**
     * dgamma = ∑delta * xhat
     * <p>
     * dbeta = ∑delta
     * <p>
     * dxhat = delta * gamma
     * <p>
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)
     * <p>
     * dx = dxhat / (var + eta)^1/2 + dvar * 2.0f * (x - mean) / N + dmean / N
     *
     * @param mean
     * @param var
     * @param xhat
     * @param delta
     * @param gamma
     * @param diff
     * @param type
     */
    public void backward_cpu(float[][][][] input, float[] mean, float[] var, float[][][][] xhat, float[][][][] delta, float[] gamma, float[][][][] diff, float[] dgama, float[] dbeta, int type) {
        int S = W;
        if (type == 1) {
            S = C;
        }
        /**
         * dgamma = delta * xhat

         */
        float[] dgamma = new float[S];
        dgamma_cpu(delta, xhat, dgamma, type);
        float[] dgamma2 = new float[S];
        dgamma2_cpu(delta, input, mean, var, dgamma2, type);
        /**
         * dxhat = delta * gamma

         */
        float[][][][] dxhat = new float[N][C][H][W];
        dxhat_cpu(dxhat, delta, gamma, type);
        /**
         * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2

         */
        float[] dvar = new float[S];
        dvar_cpu(dvar, mean, var, input, dxhat, type);
        /**
         * (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)

         */
        float[] dmean = new float[S];
        dmean_cpu(dmean, mean, var, dvar, input, dxhat, type);
        System.out.println("dvar-cpu:");
        PrintUtils.printImage(dvar);
        System.out.println("");
        System.out.println("dmean-cpu:");
        PrintUtils.printImage(dmean);
        System.out.println("");
        System.out.println("dgamma-cpu:");
        PrintUtils.printImage(dgamma);
        System.out.println("");
        System.out.println("dgamma2-cpu:");
        PrintUtils.printImage(dgamma2);
        System.out.println("");
        /**
         * dx = dxhat / (var + eta)^1/2 + dvar * 2.0f * (x - mean) / N + dmean / N

         */
        //    	float[][][][] dx = new float[N][C][H][W];
        dx_cpu(diff, dxhat, mean, var, dmean, dvar, input, type);
        float[][][][] dx = new float[N][C][H][W];
        dx2_cpu(dx, delta, mean, var, gamma, input, type);
        System.out.println("dx-cpu:");
        PrintUtils.printImage(diff);
        System.out.println("");
        System.out.println("dx-cpu2:");
        PrintUtils.printImage(dx);
        System.out.println("");
        //
        //    	System.out.println("=================dx-cpu==================================");
        //    	PrintUtils.printImage(dx);
        //    	System.out.println("=================dx-cpu==================================");
    }

    /**
     * dgamma = ∑delta * xhat
     */
    private void dgamma_cpu(float[][][][] delta, float[][][][] xhat, float[] dgamma, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                dgamma[c] = 0;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dgamma[c] += delta[n][c][h][w] * xhat[n][c][h][w];
                        }
                    }
                }
            }
        } else {
            for (int w = 0; w < W; w++) {
                dgamma[w] = 0;
                for (int n = 0; n < N; n++) {
                    System.out.println(xhat[n][0][0][w] + ":" + n + ":" + w);
                    dgamma[w] += delta[n][0][0][w] * xhat[n][0][0][w];
                }
            }
        }
    }

    /**
     * dgamma = ∑delta * xhat
     */
    private void dgamma2_cpu(float[][][][] delta, float[][][][] input, float[] mean, float[] var, float[] dgamma, int type) {
        if (type == 1) {
            float[][][][] tmp = new float[N][C][H][W];
            for (int c = 0; c < C; c++) {
                dgamma[c] = 0;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            tmp[n][c][h][w] = (input[n][c][h][w] - mean[c]) * delta[n][c][h][w];
                            dgamma[c] += (input[n][c][h][w] - mean[c]) * delta[n][c][h][w];
                        }
                    }
                }
                dgamma[c] *= 1.0f / Math.sqrt(var[c] + eta);
                //System.out.println(dgamma[c]);
            }
            //    		PrintUtils.printImage(tmp);
            float tmp2 = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            tmp2 += tmp[n][c][h][w];
                        }
                    }
                }
            }
            System.out.println(tmp2);
        } else {
            for (int w = 0; w < W; w++) {
                dgamma[w] = 0;
                for (int n = 0; n < N; n++) {
                    dgamma[w] += (input[n][0][0][w] - mean[w]) * delta[n][0][0][w];
                }
                dgamma[w] /= Math.sqrt(var[w] + eta);
            }
        }
    }

    /**
     * dxhat = delta * gamma
     *
     * @param dxhat
     * @param delta
     * @param gamma
     * @param type
     */
    private void dxhat_cpu(float[][][][] dxhat, float[][][][] delta, float[] gamma, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dxhat[n][c][h][w] = delta[n][c][h][w] * gamma[c];
                        }
                    }
                }
            }
        } else {
            for (int w = 0; w < W; w++) {
                for (int n = 0; n < N; n++) {
                    dxhat[n][0][0][w] = delta[n][0][0][w] * gamma[w];
                }
            }
        }
    }

    /**
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     */
    private void dvar_cpu(float[] dvar, float[] mean, float[] var, float[][][][] input, float[][][][] dxhat, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                float val = 0.0f;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            val += dxhat[n][c][h][w] * (input[n][c][h][w] - mean[c]) * -0.5f;
                        }
                    }
                }
                dvar[c] = (float) (val * Math.pow((var[c] + eta), -1.5f));
            }
        } else {
            for (int w = 0; w < W; w++) {
                float val = 0.0f;
                for (int n = 0; n < N; n++) {
                    val += dxhat[n][0][0][w] * (input[n][0][0][w] - mean[w]) * -0.5f;
                }
                dvar[w] = (float) (val * Math.pow((var[w] + eta), -1.5f));
            }
        }
    }

    /**
     * (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)
     */
    private void dmean_cpu(float[] dmean, float[] mean, float[] var, float[] dvar, float[][][][] input, float[][][][] dxhat, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                dmean[c] = 0.0f;
                float tmp1 = 0.0f;
                float tmp2 = 0.0f;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            tmp1 += -1.0f * dxhat[n][c][h][w] / Math.sqrt(var[c] + eta);
                            tmp2 += dvar[c] * -2.0f / (N * H * W) * (input[n][c][h][w] - mean[c]);
                        }
                    }
                }
                dmean[c] = tmp1 + tmp2;
            }
        } else {
            for (int w = 0; w < W; w++) {
                dmean[w] = 0.0f;
                float tmp1 = 0.0f;
                float tmp2 = 0.0f;
                for (int n = 0; n < N; n++) {
                    tmp1 += -1.0f * dxhat[n][0][0][w];
                    tmp2 += -2.0f * (input[n][0][0][w] - mean[w]);
                }
                dmean[w] = (float) (tmp1 / Math.sqrt(var[w] + eta) + dvar[w] * tmp2 / N);
            }
        }
    }

    /**
     * dx = dxhat / (var + eta)^1/2 + dvar * 2.0f * (x - mean) / N + dmean / N
     */
    private void dx_cpu(float[][][][] dx, float[][][][] dxhat, float[] mean, float[] var, float[] dmean, float[] dvar, float[][][][] input, int type) {
        if (type == 1) {
            for (int c = 0; c < C; c++) {
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dx[n][c][h][w] = (float) (dxhat[n][c][h][w] / Math.sqrt(var[c] + eta) + dvar[c] * 2.0f * (input[n][c][h][w] - mean[c]) / (N * H * W) + dmean[c] / (N * H * W));
                        }
                    }
                }
            }
        } else {
            for (int w = 0; w < W; w++) {
                for (int n = 0; n < N; n++) {
                    dx[n][0][0][w] = (float) (dxhat[n][0][0][w] / Math.sqrt(var[w] + eta) + dvar[w] * 2.0f * (input[n][0][0][w] - mean[w]) / N + dmean[w] / N);
                }
            }
        }
    }

    /**
     * dx = dxhat / (var + eta)^1/2 + dvar * 2.0f * (x - mean) / N + dmean / N
     */
    private void dx2_cpu(float[][][][] dx, float[][][][] delta, float[] mean, float[] var, float[] gamma, float[][][][] input, int type) {
        if (type == 1) {
            float[] dot_p = new float[mean.length];
            float[] gmean = new float[mean.length];
            float[] proj_scale = new float[mean.length];
            float[] grad_scale = new float[mean.length];
            float[] invstd = new float[mean.length];
            float norm = 1.0f / N / H / W;
            for (int c = 0; c < C; c++) {
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dot_p[c] += (input[n][c][h][w] - mean[c]) * delta[n][c][h][w];
                            gmean[c] += delta[n][c][h][w] * norm;
                        }
                    }
                }
            }
            for (int c = 0; c < C; c++) {
                invstd[c] = 1.0f / (float) Math.sqrt(var[c] + eta);
                proj_scale[c] = dot_p[c] * norm * invstd[c] * invstd[c];
                grad_scale[c] = invstd[c] * gamma[c];
            }
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            float proj = (input[n][c][h][w] - mean[c]) * proj_scale[c];
                            dx[n][c][h][w] = (delta[n][c][h][w] - gmean[c] - proj) * grad_scale[c];
                        }
                    }
                }
            }
        } else {
            float[] dot_p = new float[mean.length];
            float[] gmean = new float[mean.length];
            float[] proj_scale = new float[mean.length];
            float[] grad_scale = new float[mean.length];
            float[] invstd = new float[mean.length];
            float norm = 1.0f / N;
            for (int w = 0; w < W; w++) {
                for (int n = 0; n < N; n++) {
                    dot_p[w] += (input[n][0][0][w] - mean[w]) * delta[n][0][0][w];
                    gmean[w] += delta[n][0][0][w] * norm;
                }
            }
            for (int w = 0; w < W; w++) {
                invstd[w] = 1.0f / (float) Math.sqrt(var[w] + eta);
                proj_scale[w] = dot_p[w] * norm * invstd[w] * invstd[w];
                grad_scale[w] = invstd[w] * gamma[w];
            }
            for (int n = 0; n < N; n++) {
                for (int w = 0; w < W; w++) {
                    float proj = (input[n][0][0][w] - mean[w]) * proj_scale[w];
                    dx[n][0][0][w] = (delta[n][0][0][w] - gmean[w] - proj) * grad_scale[w];
                }
            }
        }
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
                float dmu_sum = 0.0f;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dvar_val += (x[n][c][h][w] - mean[c]) * dz[n][c][h][w];
                            dmu_val += -1.0f * dz[n][c][h][w] / (float) Math.sqrt(var[c] + eta);
                            dmu_sum += -2.0f * (x[n][c][h][w] - mean[c]);
                        }
                    }
                }
                dvar[c] = (float) (dvar_val * -0.5f * Math.pow(var[c] + eta, -1.5));
                dmu[c] = dmu_val + dvar[c] * dmu_sum / N / H / W;
            }
        } else {
            for (int w = 0; w < W; w++) {
                float dvar_val = 0.0f;
                float dmu_val = 0.0f;
                float dmu_sum = 0.0f;
                for (int n = 0; n < N; n++) {
                    dvar_val += (x[n][0][0][w] - mean[w]) * dz[n][0][0][w];
                    dmu_val += -1.0f * dz[n][0][0][w] / (float) Math.sqrt(var[w] + eta);
                    dmu_sum += -2.0f * (x[n][0][0][w] - mean[w]);
                }
                dvar[w] = (float) (dvar_val * -0.5f * Math.pow(var[w] + eta, -1.5));
                dmu[w] = dmu_val + dvar[w] * dmu_sum / N;
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
                        dx[n][c][h][w] = (sn * dxhat[n][c][h][w] - dxhatSum[c] + dxhat_Sum[c] * dxhatSum[c] / sn - xhat[n][c][h][w] * dxhat_Sum[c]) / (sn * std[c]);
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
            float m1 = dxhatSum[c] / sn;
            float m2 = dxhat_Sum[c] / sn;
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

