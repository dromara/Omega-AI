package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class FlashAttentionKernel2 extends BaseKernel {
    private CUfunction forward_function;
    private CUfunction backward_function;
    private Pointer kernelParameters;
    
    private Tensor row_max;
    private Tensor row_sum;

    public FlashAttentionKernel2(int time, int headDim, CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String[] args) {
        int batchSize = 2;
        int time = 16;
        int headDim = 64; //headDim
        int len = batchSize * time * headDim;
        Tensor Q = new Tensor(batchSize, time, 1, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        Tensor K = new Tensor(batchSize, time, 1, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        Tensor V = new Tensor(batchSize, time, 1, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
//        Tensor Q = new Tensor(batchSize, time, 1, headDim, MatrixUtils.order(len, 0.01f, 0.01f), true);
//        Tensor K = new Tensor(batchSize, time, 1, headDim, MatrixUtils.order(len, 0.02f, 0.02f), true);
//        Tensor V = new Tensor(batchSize, time, 1, headDim, MatrixUtils.order(len, 0.03f, 0.03f), true);
        Tensor dQ = new Tensor(batchSize, time, 1, headDim, true);
        Tensor dK = new Tensor(batchSize, time, 1, headDim, true);
        Tensor dV = new Tensor(batchSize, time, 1, headDim, true);
        Tensor output = new Tensor(batchSize, time, 1, headDim, true);
        Tensor delta = new Tensor(batchSize, time, 1, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        CUDAManager cudaManager = new CUDAManager(0);
        FlashAttentionKernel2 kernel = new FlashAttentionKernel2(time, headDim, cudaManager);
        for (int i = 0; i < 20; i++) {
            long startTime = System.nanoTime();
            kernel.forward(Q, K, V, output);
            JCuda.cudaDeviceSynchronize();
//            output.showDMByOffset(0, 100);
//            output.showDM();
            System.out.println((System.nanoTime() - startTime) / 1e6 + "ms.");
            //			output.
            long startTime2 = System.nanoTime();
            kernel.backward(Q, K, V, output, delta, dQ, dK, dV);
            JCuda.cudaDeviceSynchronize();
            System.out.println((System.nanoTime() - startTime2) / 1e6 + "ms.");
//            			dQ.showDMByOffset(0, 100);
//            			dK.showDMByOffset(0, 100);
//            			dV.showDMByOffset(0, 100);
            		dQ.showDM();
            			dV.showDM();
        }
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
        float[] S = new float[Bc * Br];
        for (int Tr_i = 0; Tr_i < Tr; Tr_i++) {
            for (int tx = 0; tx < d; tx++) {
                if (tx < d) {
                    for (int i = 0; i < Br; i++) {
                        Qi[i * d + tx] = Q[Tr_i * Br * d + i * d + tx];
                    }
                }
            }
            System.out.println(JsonUtils.toJson(Qi));
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
                System.out.println(JsonUtils.toJson(Kj));
                mul_kA_BT(Qi, Kj, S, Br, Bc, d, Bc);
                System.out.println(JsonUtils.toJson(S));
            }
        }
    }

    public static void mul_kA_BT(float[] A, float[] B, float[] C, int m, int n, int k, int d) {
        for (int tx = 0; tx < d; tx++) {
            for (int y = 0; y < n; y++) {
                float sum = 0;
                for (int x = 0; x < k; x++) {
                    sum += A[(tx * k) + x] * B[(y * k) + x];
                }
                C[(n * tx) + y] = sum;
            }
        }
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("FlashAttentionForwardKernel.cu", "forward_attention_kernel");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("FlashAttentionBackwardKernel.cu", "backward_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void initParam(int B,int N) {
    	if(row_max == null || B != row_max.number) {
    		row_max = new Tensor(B, 1, 1, N, true);
    		row_sum = new Tensor(B, 1, 1, N, true);
    	}
    	this.fill_gpu(row_max, Float.NEGATIVE_INFINITY);
    	this.fill_gpu(row_sum, 0);
    }
    
    public void forward(Tensor Q, Tensor K, Tensor V, Tensor output) {
        try {
           
        	int num_threads_per_block = 1024;
        	int B = Q.number;
            int N = Q.channel;
            int D = Q.height * Q.width;
            
//            long startTime = System.nanoTime();
            
            initParam(B, N);
//            JCuda.cudaDeviceSynchronize();
//            System.out.println("init:"+(System.nanoTime() - startTime) / 1e6 + "ms.");
            
            int num_rows_per_block = num_threads_per_block / D; 
            int num_blocks_per_sample = (int) Math.ceil(N / num_rows_per_block);
        	
            int[] grid = new int[] {B, 1, 1};
            int BLOCK_SIZE = num_rows_per_block * D;
            
            if(num_rows_per_block * D <= num_rows_per_block * num_rows_per_block + 3 * num_rows_per_block) {
            	throw new RuntimeException("num_rows_per_block * D must be than num_rows_per_block * num_rows_per_block + 3 * num_rows_per_block.");
            }
            
            if(BLOCK_SIZE > num_threads_per_block) {
            	throw new RuntimeException("BLOCK_SIZE > num_threads_per_block.");
            }
//            System.out.println(num_rows_per_block);
            if(num_rows_per_block > D){
            	throw new RuntimeException("num_rows_per_block > D.");
            }
            
            if(D > num_threads_per_block){
            	throw new RuntimeException("num_threads_per_block <= D.");
            }
            
            /**
             *  float* query,
			    float* key,
			    float* value,
			    float* outputs,
			    float* rowmax_statistics, 
			    float* rowsum_statistics,
			    int batch_size, int sequence_length, int dimension,
			    int block_size,
			    int num_rows_per_block,
			    int num_blocks_per_sample
             */
            kernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(row_max.getGpuData()), Pointer.to(row_sum.getGpuData()),
            		Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{D}),
            		Pointer.to(new int[]{BLOCK_SIZE}), Pointer.to(new int[]{num_rows_per_block}), Pointer.to(new int[]{num_blocks_per_sample}));
            
            int sram_size = 3 * BLOCK_SIZE * Sizeof.FLOAT;
            
            checkCUDA(cuLaunchKernel(forward_function, grid[0], grid[1], grid[2],      // Grid dimension
            		num_threads_per_block, 1, 1,      // Block dimension
                    sram_size, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
                    ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor Q, Tensor K, Tensor V, Tensor output, Tensor delta, Tensor dQ, Tensor dK, Tensor dV) {
        try {
        	
        	int num_threads_per_block = 1024;
        	int B = Q.number;
            int N = Q.channel;
            int D = Q.height * Q.width;
            
            int num_rows_per_block = num_threads_per_block / D; 
            int num_rows_per_block_squared = num_rows_per_block * num_rows_per_block;
            int BLOCK_SIZE = num_rows_per_block * D;
            int num_blocks_per_sample = N / num_rows_per_block;
            
            int[] grid = new int[] {B, 1, 1};

            /**
             * float* query,
		    float* key,
		    float* value,
		    float* outputs,
		    float* d_outputs,
		    float* rowmax_statistics, 
		    float* rowsum_statistics,
		    float* d_query, 
		    float* d_key,
		    float* d_value,
		    int batch_size, int sequence_length, int dimension,
		    int block_size,
		    int num_rows_per_block,
		    int num_blocks_per_sample
             */
            kernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(output.getGpuData()),Pointer.to(delta.getGpuData()),
            		Pointer.to(row_max.getGpuData()), Pointer.to(row_sum.getGpuData()),
            		Pointer.to(dQ.getGpuData()), Pointer.to(dK.getGpuData()),Pointer.to(dV.getGpuData()),
            		Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{D}),
            		Pointer.to(new int[]{BLOCK_SIZE}), Pointer.to(new int[]{num_rows_per_block}), Pointer.to(new int[]{num_blocks_per_sample}));
            
            int sram_size = (5 * BLOCK_SIZE + 3 * num_rows_per_block_squared) * Sizeof.FLOAT;
            
            checkCUDA(cuLaunchKernel(backward_function, grid[0], grid[1], grid[2],      // Grid dimension
            		num_threads_per_block, 1, 1,      // Block dimension
                    sram_size, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
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

