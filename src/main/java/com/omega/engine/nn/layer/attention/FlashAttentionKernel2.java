package com.omega.engine.nn.layer.attention;

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
    private CUfunction d_backward_function;
    private CUfunction backward_function;
    private Pointer kernelParameters;
    
    private Tensor D;

    public FlashAttentionKernel2(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String[] args) {
        int batchSize = 32;
        int headNum = 16;
        int time = 256;
        int headDim = 64; //headDim
        int len = batchSize * headNum * time * headDim;
        Tensor Q = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        Tensor K = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        Tensor V = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
        
//        String qPath = "D:\\models\\dit_q.json";
//	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(qPath);
//	    ModeLoaderlUtils.loadData(Q, datas, "q");
//	    
//        String kPath = "D:\\models\\dit_k.json";
//	    Map<String, Object> kdatas = LagJsonReader.readJsonFileSmallWeight(kPath);
//	    ModeLoaderlUtils.loadData(K, kdatas, "k");
//	    
//        String vPath = "D:\\models\\dit_v.json";
//	    Map<String, Object> vdatas = LagJsonReader.readJsonFileSmallWeight(vPath);
//	    ModeLoaderlUtils.loadData(V, vdatas, "v");
        
        Tensor dQ = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dK = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dV = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor log = new Tensor(batchSize, headNum, 1, time, true);
        Tensor output = new Tensor(batchSize, headNum, time, headDim, true);
        
        Tensor delta = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
//        String dPath = "D:\\models\\dit_attn_delta.json";
//	    Map<String, Object> ddatas = LagJsonReader.readJsonFileSmallWeight(dPath);
//	    ModeLoaderlUtils.loadData(delta, ddatas, "delta");
        
        CUDAManager cudaManager = new CUDAManager(0);
        
        FlashAttentionKernel2 kernel = new FlashAttentionKernel2(cudaManager);

        for (int i = 0; i < 20; i++) {
            long startTime = System.nanoTime();
            kernel.forward(Q, K, V, output, log, batchSize, headNum, time, headDim);
            JCuda.cudaDeviceSynchronize();
//            output.showDMByOffset(0, 100);
//            output.showDM();
            System.out.println((System.nanoTime() - startTime) / 1e6 + "ms.");
//            //			output.
            long startTime2 = System.nanoTime();
            kernel.backward(Q, K, V, output, log, delta, dQ, dK, dV, batchSize, headNum, time, headDim);
            JCuda.cudaDeviceSynchronize();
            System.out.println((System.nanoTime() - startTime2) / 1e6 + "ms.");
////            			dQ.showDMByOffset(0, 100);
////            			dK.showDMByOffset(0, 100);
////            			dV.showDMByOffset(0, 100);
//            		dQ.showDM();
//            			dV.showDM();
        }
//        output.showDM("output");
//        dQ.showDM("dQ");
//        dK.showDM("dK");
//        dV.showDM("dV");
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
                forward_function = getCudaManager().getLocalFunctionByModule("flash_attention_kernel2.cu", "flash_attention2_forward_kernel_optim");
            }
            if (d_backward_function == null) {
            	d_backward_function = getCudaManager().getLocalFunctionByModule("flash_attention_backward_kernel2.cu", "D_computation_reduction_kernel");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("flash_attention_backward_kernel2.cu", "flash_attention2_backward_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public int next_power_of_2(int x) {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x++;
        return x;
    }
    
    public void backward(Tensor Q, Tensor K, Tensor V, Tensor output, Tensor logsumexp, Tensor delta, Tensor dQ, Tensor dK, Tensor dV, int B, int HN, int SQ, int HD) {
        try {
        	
        	int BLOCK_SIZE_C = 32;
            int BLOCK_SIZE_R = 32;
            
            D = getD(B, HN, SQ);
            
            /**
             *  const float *d_output, 
			    const float *output, 
			    int batch_size, 
			    int num_heads, 
			    int seq_len,
			    int head_dim,
			    float *D
             */
            Pointer kernelParameters1 = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{B}), Pointer.to(new int[]{HN}), Pointer.to(new int[]{SQ}),  Pointer.to(new int[]{HD}),
            		Pointer.to(D.getGpuData()));
            
            int threads_per_block_di = next_power_of_2(HD);
            int total_blocks_di = B * HN * SQ;
            int shared_mem_size_di = Sizeof.FLOAT * threads_per_block_di;
            
            checkCUDA(cuLaunchKernel(d_backward_function, total_blocks_di, 1, 1,      // Grid dimension
            		threads_per_block_di, 1, 1,      // Block dimension
            		shared_mem_size_di, null,               // Shared memory size and stream
                    kernelParameters1, null // Kernel- and extra parameters
                    ));
        	
            int T_r = (SQ + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
            int T_c = (SQ + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;
            int total_blocks = B * HN * T_c;
            int num_threads_per_block = 256;
            
            int q_buff = BLOCK_SIZE_R * HD; // for q tile and d_o tile
            int k_buff = BLOCK_SIZE_C * HD;
            int v_buff = BLOCK_SIZE_C * HD;
            int d_k_buff = BLOCK_SIZE_C * HD;
            int d_v_buff = BLOCK_SIZE_C * HD;
            int logsumexp_buff = BLOCK_SIZE_R; // for logsumexp_i and d_i
            int p_buff = BLOCK_SIZE_R * BLOCK_SIZE_C; // to store P_ij and dP_ij and S_ij
            
            int shared_mem_size = (q_buff + k_buff + v_buff + d_k_buff + d_v_buff + logsumexp_buff + p_buff) * Sizeof.FLOAT;
            
            /**
             *  const float* __restrict__ query,
			    const float* __restrict__ key,
			    const float* __restrict__ value,
			    const float* __restrict__ output,
			    const float* __restrict__ d_output,
			    const float* __restrict__ logsumexp,
			    const float* __restrict__ d,
			    float* __restrict__ d_query,
			    float* __restrict__ d_key,
			    float* __restrict__ d_value,
			    int batch_size,
			    int num_heads,
			    int seq_len,
			    int head_dim
             */
            dQ.clearGPU();
//            dK.clearGPU();
//            dV.clearGPU();
            Pointer kernelParameters2 = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(delta.getGpuData()), Pointer.to(logsumexp.getGpuData()), Pointer.to(D.getGpuData()),
            		Pointer.to(dQ.getGpuData()), Pointer.to(dK.getGpuData()), Pointer.to(dV.getGpuData()),
            		Pointer.to(new int[]{B}), Pointer.to(new int[]{HN}), Pointer.to(new int[]{SQ}),  Pointer.to(new int[]{HD}));
            
            checkCUDA(cuLaunchKernel(backward_function, total_blocks, 1, 1,      // Grid dimension
            		num_threads_per_block, 1, 1,      // Block dimension
            		shared_mem_size, null,               // Shared memory size and stream
            		kernelParameters2, null // Kernel- and extra parameters
                    ));
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Tensor Q, Tensor K, Tensor V, Tensor output, Tensor logsumexp, int B, int NH, int SQ, int HD) {
        try {
        	
            /**
             *  const float* __restrict__ query,
			    const float* __restrict__ key,
			    const float* __restrict__ value,
			    float* __restrict__ output,
			    float* __restrict__ logsumexp,
			    int batch_size,
			    int num_heads,
			    int seq_len,
			    int head_dim
             */
            kernelParameters = Pointer.to(Pointer.to(Q.getGpuData()), Pointer.to(K.getGpuData()), Pointer.to(V.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(logsumexp.getGpuData()),
            		Pointer.to(new int[]{B}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{SQ}), Pointer.to(new int[]{HD}));

            int BLOCK_SIZE_C = 32;
            int BLOCK_SIZE_R = 32;
            int BK = 4;             // tile size for head_dim dimension
            int TM = 4;             // each thread handles TM rows
            int TN = 4;             // each thread handles TN cols
            
            int T_r = (SQ + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
            int total_blocks = B * NH * T_r;
            // const size_t num_threads_per_block = (BLOCK_SIZE_R * BLOCK_SIZE_C) / (TM * TN);
            int num_threads_per_block = 128;
            
            int BM = BLOCK_SIZE_C;
            int BN = BLOCK_SIZE_R;
            
            int q_buff = BM * HD;
            int kv_buff = BN * HD;
            int s_buff = BM * BN;
            int o_buff=  BM * HD;
            int logsumexp_buff = BM;
            int maxes = BM;
            int exp_norm_coeffs = BM;
            
            int shared_mem_size = (q_buff + kv_buff + s_buff + o_buff + logsumexp_buff + maxes + exp_norm_coeffs) * Sizeof.FLOAT;

            checkCUDA(cuLaunchKernel(forward_function, total_blocks, 1, 1,      // Grid dimension
            		num_threads_per_block, 1, 1,      // Block dimension
            		shared_mem_size, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
                    ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public Tensor getD(int batchSize, int num_heads, int seq_len) {
    	int len = batchSize * num_heads * seq_len;
    	if(D == null || D.dataLength != len) {
    		D = Tensor.createGPUTensor(D, 1, 1, 1, len, true);
    	}
    	return D;
    }
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

