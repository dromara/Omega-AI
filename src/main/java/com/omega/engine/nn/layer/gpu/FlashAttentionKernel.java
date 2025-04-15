package com.omega.engine.nn.layer.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAModules;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class FlashAttentionKernel extends BaseKernel {
    private CUfunction forward_function;
    private CUfunction forward_function2;
    private CUfunction backward_function;
    private Pointer kernelParameters;
    private Pointer bwdKernelParameters;
    private Tensor d_m;
    private Tensor d_l;
    private int headNum;
    private int time;
    private int headDim;

    public FlashAttentionKernel(int headNum, int time, int headDim, CUDAManager cudaManager) {
        super(cudaManager);
        this.headNum = headNum;
        this.time = time;
        this.headDim = headDim;
        init();
    }

    public static void main(String[] args) {
        CUDAModules.initContext();
        int batchSize = 8;
        int headNum = 16;
        int time = 512;
        int headDim = 32; //headDim
        int len = batchSize * headNum * time * headDim;
        Tensor Q = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
        Tensor K = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
        Tensor V = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
        Tensor dQ = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dK = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dV = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor output = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor delta = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
        CUDAManager cudaManager = new CUDAManager(0);
        FlashAttentionKernel kernel = new FlashAttentionKernel(headNum, time, headDim, cudaManager);
        for (int i = 0; i < 10; i++) {
            long startTime = System.nanoTime();
            kernel.forward2(Q, K, V, output);
            JCuda.cudaDeviceSynchronize();
            System.out.println((System.nanoTime() - startTime) / 1e6 + "ms.");
            //			output.showDM(0);
            //			long startTime2 = System.nanoTime();
            //			kernel.backward(Q, K, V, output, delta, dQ, dK, dV);
            //			System.out.println((System.nanoTime() - startTime2)/1e6+"ms.");
            //
            ////			dQ.showDM(0);
            //			output.clear();
            //			delta.clear();
        }
        //
        //		kernel.backward(Q, K, V, null, null, null, null);
        //		test();
        //		kernel.forward2(Q, K, V, output);
    }

    public static void test() {
        int Tr = 2;
        int Tc = 2;
        int Br = 2;
        int Bc = 2;
        int d = 3;
        int N = 4;
        int tile_size = Bc * d;
        int len = 1 * 1 * N * d;
        float[] Q = MatrixUtils.order(len, 1f, 1f);
        float[] K = MatrixUtils.order(len, 1f, 1f);
        float[] Qi = new float[Br * d];
        float[] Kj = new float[Bc * d];
        float[] S = new float[Bc * d];
        for (int j = 0; j < Tc; j++) {
            for (int tx = 0; tx < Bc; tx++) {
                // Load Kj, Vj to SRAM
                for (int x = 0; x < d; x++) {
                    if (j * Bc + tx < N) {
                        Kj[(tx * d) + x] = K[(tile_size * j) + (tx * d) + x];
                    } else {
                        Kj[(tx * d) + x] = 0;
                    }
                }
            }
            System.out.println("j:[" + j + "]Kj:" + JsonUtils.toJson(Kj));
            for (int tx = 0; tx < Bc; tx++) {
                for (int i = 0; i < Tr; i++) {
                    if ((Br * i) + tx >= N) {
                        break;
                    }
                    // Load Qi to SRAM, l and m to registers
                    for (int x = 0; x < d; x++) {
                        if (i * Br + tx < N) {
                            Qi[(tx * d) + x] = Q[(tile_size * i) + (tx * d) + x];
                        } else {
                            Qi[(tx * d) + x] = 0.0f;
                        }
                    }
                    System.out.println("i:[" + i + "]Qi:" + JsonUtils.toJson(Qi));
                    // S = QK^T, row_m = rowmax(S)
                    for (int y = 0; y < Bc; y++) {
                        float sum = 0;
                        for (int x = 0; x < d; x++) {
                            //	                	System.out.println(((tx * d) + x) + "*" + ((y * d) + x));
                            //	                	System.out.println(Qi[(tx * d) + x] + "*" + Kj[(y * d) + x]);
                            sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                        }
                        S[(Bc * tx) + y] = sum;
                        System.out.println("sum:" + sum);
                    }
                    System.out.println("S:" + JsonUtils.toJson(S));
                }
            }
        }
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    private void initKernel(Tensor input, int headNum, int time) {
        if (input.number != N) {
            N = input.number;
            /**
             * 申请向前传播参数显存

             */
            this.d_m = new Tensor(input.number, headNum, time, 1, MatrixUtils.val(input.number * headNum * time, Float.NEGATIVE_INFINITY), true);
            this.d_l = new Tensor(input.number, headNum, time, 1, true);
        }
    }

    public void initFunction() {
        try {
            //			if(forward_function == null) {
            //
            //				forward_function = CUDAModules.getLocalFunctionByModule("FlashAttentionKernel.cu", "forward_attention_kernel");
            //
            //			}
            if (forward_function2 == null) {
                forward_function2 = getCudaManager().getLocalFunctionByModule("FlashAttentionKernel.cu", "forward_kernel");
            }
            //			if(backward_function == null) {
            //
            //				backward_function = CUDAModules.getLocalFunctionByModule("FlashAttentionKernel.cu", "backward_kernel");
            //
            //			}
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Tensor Q, Tensor K, Tensor V, Tensor output) {
        try {
            initKernel(Q, headNum, time);
            int B = Q.number;
            int nh = headNum;
            int N = time;
            int d = headDim;
            long max_shared_memory = CUDAModules.props.sharedMemPerBlock;
            long max_threads_num = CUDAModules.props.maxThreadsPerBlock;
            int M = (int) (max_shared_memory / Sizeof.FLOAT);
            int Bc = (int) Math.ceil(M / (4 * d));
            int Br = (int) Math.min(Math.min(Bc, d), max_threads_num);
            int Tc = (int) Math.ceil((float) N / Bc);
            int Tr = (int) Math.ceil((float) N / Br);
            System.out.println(Bc);
            System.out.println(Br);
            System.out.println(Tc);
            System.out.println(Tr);
            /**
             * 设置入参
             * float *Q, float *K, float *V, float *O, float *l, float *m, const int N, const int d,

             const int Bc, const int Br, const int Tc, const int Tr

             */
            kernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(d_l.getGpuData()), Pointer.to(d_m.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{d}), Pointer.to(new int[]{Bc}), Pointer.to(new int[]{Br}), Pointer.to(new int[]{Tc}), Pointer.to(new int[]{Tr}));
            int[] grid_dim = new int[]{B, nh, 1};
            int[] block_dim = new int[]{Br, 1, 1};
            int shared_memory_size = Sizeof.FLOAT * ((2 * Bc * d) + (Br * d) + (Bc * Br));
            System.out.println("max share memory size:" + CUDAModules.props.sharedMemPerBlock + ".current size:" + shared_memory_size);
            checkCUDA(cuLaunchKernel(forward_function, grid_dim[0], grid_dim[1], grid_dim[2],      // Grid dimension
                    block_dim[0], block_dim[1], block_dim[2],      // Block dimension
                    shared_memory_size, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            //		    d_m.showDM();
            //		    d_l.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward2(Tensor Q, Tensor K, Tensor V, Tensor output) {
        try {
            initKernel(Q, headNum, time);
            int Bc = 32;
            int Br = 32;
            int B = Q.number;
            int nh = Q.channel;
            int N = Q.height;
            int d = Q.width;
            int Tc = (int) Math.ceil((float) N / Bc);
            int Tr = (int) Math.ceil((float) N / Br);
            float softmax_scale = (float) (1.0f / Math.sqrt(d));
            /**
             * 设置入参
             * const float* Q, const float* K, const float* V, const int N, const int d,

             const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,

             float* l, float *m, float* O

             */
            kernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{d}), Pointer.to(new int[]{Tc}), Pointer.to(new int[]{Tr}), Pointer.to(new int[]{Bc}), Pointer.to(new int[]{Br}), Pointer.to(new float[]{softmax_scale}), Pointer.to(d_l.getGpuData()), Pointer.to(d_m.getGpuData()), Pointer.to(output.getGpuData()));
            int[] grid_dim = new int[]{B, nh, 1};
            int[] block_dim = new int[]{Bc, 1, 1};
            int sram_size = (3 * Bc * d * Sizeof.FLOAT) + (Bc * Br * Sizeof.FLOAT);
            System.out.println("max share memory size:" + CUDAModules.props.sharedMemPerBlock + ".current size:" + sram_size);
            checkCUDA(cuLaunchKernel(forward_function2, grid_dim[0], grid_dim[1], grid_dim[2],      // Grid dimension
                    block_dim[0], block_dim[1], block_dim[2],      // Block dimension
                    sram_size, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            //		    d_m.showDM();
            //		    d_l.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor Q, Tensor K, Tensor V, Tensor O, Tensor delta, Tensor dQ, Tensor dK, Tensor dV) {
        try {
            //			initKernel(Q, headNum, time);
            int B = Q.number;
            int nh = headNum;
            int N = time;
            int d = headDim;
            long max_shared_memory = CUDAModules.props.sharedMemPerBlock;
            long max_threads_num = CUDAModules.props.maxThreadsPerBlock;
            int block_size = 16;
            int Bc = block_size;
            int Br = block_size;
            int Tc = (int) Math.ceil((float) N / Bc);
            int Tr = (int) Math.ceil((float) N / Br);
            System.out.println(Bc);
            System.out.println(Br);
            System.out.println(Tc);
            System.out.println(Tr);
            /**
             * 设置入参
             * float *Q, float *K, float *V, float *O, float *dQ, float *dK, float *dV, float *dO,

             float *l, float *m, const int N, const int d, const int Tc, const int Tr

             */
            bwdKernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(O.getGpuData()), Pointer.to(dQ.getGpuData()), Pointer.to(dK.getGpuData()), Pointer.to(dV.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(d_l.getGpuData()), Pointer.to(d_m.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{d}), Pointer.to(new int[]{Tc}), Pointer.to(new int[]{Tr}));
            int shared_memory_size = Sizeof.FLOAT * ((7 * block_size * d) + (2 * block_size * block_size));
            int[] grid_dim = new int[]{B, nh, 1};
            int[] block_dim = new int[]{block_size, 1, 1};
            System.out.println("max share memory size:" + CUDAModules.props.sharedMemPerBlock + ".current size:" + shared_memory_size);
            checkCUDA(cuLaunchKernel(backward_function, grid_dim[0], grid_dim[1], grid_dim[2],      // Grid dimension
                    block_dim[0], block_dim[1], block_dim[2],      // Block dimension
                    shared_memory_size, null,               // Shared memory size and stream
                    bwdKernelParameters, null // Kernel- and extra parameters
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
}

