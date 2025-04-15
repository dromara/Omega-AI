package com.omega.engine.updater.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.Network;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class RMSPropKernel extends BaseKernel {
    public Tensor rw;
    public Tensor rb;
    private float mua = 0.9f;
    private float eta = 1e-12f;
    private CUfunction function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private int clamp = 0;
    private float min = -0.01f;
    private float max = 0.01f;
    private float weight_decay = 0.0005f;

    public RMSPropKernel(int weightLength, CUDAManager cudaManager) {
        super(cudaManager);
        this.rw = new Tensor(1, 1, 1, weightLength, true);
        init();
    }

    public RMSPropKernel(int weightLength, int biasLength, CUDAManager cudaManager) {
        super(cudaManager);
        this.rw = new Tensor(1, 1, 1, weightLength, true);
        this.rb = new Tensor(1, 1, 1, biasLength, true);
        init();
    }

    public RMSPropKernel(int weightLength, int clamp, float min, float max, CUDAManager cudaManager) {
        super(cudaManager);
        this.rw = new Tensor(1, 1, 1, weightLength, true);
        this.clamp = clamp;
        this.min = min;
        this.max = max;
        init();
    }

    public RMSPropKernel(int weightLength, int biasLength, int clamp, float min, float max, CUDAManager cudaManager) {
        super(cudaManager);
        this.rw = new Tensor(1, 1, 1, weightLength, true);
        this.rb = new Tensor(1, 1, 1, biasLength, true);
        this.clamp = clamp;
        this.min = min;
        this.max = max;
        init();
    }

    public static void main(String args[]) {
        CUDAManager cudaManager = new CUDAManager(0);
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 8;
        float[] test = new float[]{0.0075240037f, 0.022312285f, 0.037100658f, 0.05188888f, 0.06667703f, 0.08146531f, 0.09625361f, 0.111041375f, 0.12582973f, 0.14061777f, 0.15540561f, 0.17019409f, 0.18498187f, 0.19977006f, 0.21455756f, 0.22934535f, 0.24413382f, 0.25892144f, 0.27370873f, 0.2884969f, 0.303285f, 0.3180732f, 0.33286023f, 0.347648f, 0.36243534f, 0.37722275f, 0.39201018f, 0.40679908f, 0.42158592f, 0.43637308f, 0.45116156f, 0.46594855f, 0.48073593f, 0.4955238f, 0.5103103f, 0.52509815f, 0.53988534f, 0.55467236f};
        float[] x1 = new float[test.length];
        float[] bias1 = RandomUtils.order(N * C * H * W, 0.00001f, 0.00001f);
        Tensor w = new Tensor(1, 1, 1, test.length, x1, true);
        Tensor delta = new Tensor(1, 1, 1, test.length, test, true);
        BPNetwork net = new BPNetwork(new SoftmaxWithCrossEntropyLoss());
        net.train_time = 1;
        net.number = N;
        RMSPropKernel k = new RMSPropKernel(bias1.length, cudaManager);
        delta.showDM();
        for (int i = 0; i < 500; i++) {
            k.updateW(delta, w, net, 0.0001f);
            w.showDM();
        }
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
                function = getCudaManager().getLocalFunctionByModule("updater.cu", "RMSProp");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void updateW(Tensor diffW, Tensor weight, Network net, float lr) {
        try {
            /**
             * 设置入参
             * float *diffW, float *rw, float *weight, float mul, float eta, float learnRate, int n, int batch, int clamp, float min, float max

             */
            kernelParameters = Pointer.to(Pointer.to(diffW.getGpuData()), Pointer.to(rw.getGpuData()), Pointer.to(weight.getGpuData()), Pointer.to(new float[]{mua}), Pointer.to(new float[]{eta}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{weight_decay}), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new int[]{net.number}), Pointer.to(new int[]{clamp}), Pointer.to(new float[]{min}), Pointer.to(new float[]{max}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateW(Tensor diffW, Tensor weight, Network net, float lr, int batchSize) {
        try {
            /**
             * 设置入参
             * float *diffW, float *rw, float *weight, float mul, float eta, float learnRate, int n, int batch, int clamp, float min, float max

             */
            kernelParameters = Pointer.to(Pointer.to(diffW.getGpuData()), Pointer.to(rw.getGpuData()), Pointer.to(weight.getGpuData()), Pointer.to(new float[]{mua}), Pointer.to(new float[]{eta}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{weight_decay}), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new int[]{batchSize}), Pointer.to(new int[]{clamp}), Pointer.to(new float[]{min}), Pointer.to(new float[]{max}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateB(Tensor diffBias, Tensor bias, Network net, float lr) {
        try {
            /**
             * 设置入参
             * float *diffW, float *rw, float *weight, float mul, float eta, float learnRate, int n, int batch, int clamp, float min, float max

             */
            kernelParameters = Pointer.to(Pointer.to(diffBias.getGpuData()), Pointer.to(rb.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new float[]{mua}), Pointer.to(new float[]{eta}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{weight_decay}), Pointer.to(new int[]{diffBias.dataLength}), Pointer.to(new int[]{net.number}), Pointer.to(new int[]{clamp}), Pointer.to(new float[]{min}), Pointer.to(new float[]{max}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffBias.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateB(Tensor diffBias, Tensor bias, Network net, float lr, int batchSize) {
        try {
            /**
             * 设置入参
             * float *diffW, float *rw, float *weight, float mul, float eta, float learnRate, int n, int batch, int clamp, float min, float max

             */
            kernelParameters = Pointer.to(Pointer.to(diffBias.getGpuData()), Pointer.to(rb.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new float[]{mua}), Pointer.to(new float[]{eta}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{weight_decay}), Pointer.to(new int[]{diffBias.dataLength}), Pointer.to(new int[]{batchSize}), Pointer.to(new int[]{clamp}), Pointer.to(new float[]{min}), Pointer.to(new float[]{max}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffBias.dataLength), 1, 1,      // Grid dimension
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

