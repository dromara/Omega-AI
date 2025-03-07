package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class FlashAttentionV2Kernel extends BaseKernel{

	private CUfunction forward_function;
	
	private CUfunction backward_function;
	
	private Pointer kernelParameters;
	
	private Pointer backwardKernelParameters;

	private Tensor d_l;
	
	private int headNum;
	
	private int time;
	
	private int headDim;
	
	public FlashAttentionV2Kernel(int headNum,int time,int headDim,CUDAManager cudaManager) {
		super(cudaManager);
		this.headNum = headNum;
		this.time = time;
		this.headDim = headDim;
		init();
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();

	}
	
	private void initKernel(Tensor input,int headNum,int time) {
		
		if(input.number != N) {
			N = input.number;
			/**
			 * 申请向前传播参数显存
			 */

			this.d_l = new Tensor(input.number, headNum, time, 1, true);

		}
		
	}
	
	public void initFunction() {
		
		try {

			if(forward_function == null) {

				forward_function = getCudaManager().getLocalFunctionByModule("FlashAttentionKernel2.cu", "flash_attention_2_forward_kernel");
				
			}

			if(backward_function == null) {

				backward_function = getCudaManager().getLocalFunctionByModule("FlashAttentionKernel2.cu", "flash_attention_2_backward_kernel");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward(Tensor Q,Tensor K,Tensor V,Tensor output) {
		
		try {
			
			initKernel(Q, headNum, time);
			
			int B = Q.number;
			int nh = headNum;
			int N = time;
			int d = headDim;
			
			int Bc = 32;
			int Br = 32;
			
			int Tc = (int) Math.ceil((float) N / Bc);
			int Tr = (int) Math.ceil((float) N / Br);
			float scale = (float) (1.0f / Math.sqrt(d));
			
//			System.out.println(Bc);
//			System.out.println(Br);
//			System.out.println(Tc);
//			System.out.println(Tr);
	        /**
	         *  设置入参
	         *  const float* Q,
			    const float* K,
			    const float* V,
			    const int N,
			    const int d,
			    const int Tc,
			    const int Tr,
			    const int Bc,
			    const int Br,
			    const float softmax_scale,
			    float* L,
			    float* O
	         */ 

			kernelParameters = Pointer.to(
	        		Pointer.to(Q.getGpuData()),
	        		Pointer.to(K.getGpuData()),
	        		Pointer.to(V.getGpuData()),
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(new int[]{d}),
	        		Pointer.to(new int[]{Tc}),
	        		Pointer.to(new int[]{Tr}),
	        		Pointer.to(new int[]{Bc}),
	        		Pointer.to(new int[]{Br}),
	        		Pointer.to(new float[]{scale}),
	        		Pointer.to(d_l.getGpuData()),
	        		Pointer.to(output.getGpuData())
	            );
	        
			int col_tile_size = Bc * d;  // size of Kj, Vj
		    int row_tile_size = Br * d;  // size of Qi
		    int sram_size = (col_tile_size * 2 * Sizeof.FLOAT) + (row_tile_size * Sizeof.FLOAT) + (Bc * Br * Sizeof.FLOAT);

			int[] grid_dim = new int[] {B, nh, 1};
			int[] block_dim = new int[] {Br, 1, 1};
			
		    checkCUDA(cuLaunchKernel(forward_function,
		    		grid_dim[0],  grid_dim[1], grid_dim[2],      // Grid dimension
		    		block_dim[0], block_dim[1], block_dim[2],      // Block dimension
		    		sram_size, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

//		    d_l.showDM();
//		    JCuda.cudaDeviceSynchronize();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor Q,Tensor K,Tensor V,Tensor output,Tensor delta,Tensor dQ,Tensor dK,Tensor dV) {
		
		try {
			
			int B = Q.number;
			int nh = headNum;
			int N = time;
			int d = headDim;

			int Bc = 16;
			int Br = 16;
			
			int Tc = (int) Math.ceil((float) N / Bc);
			int Tr = (int) Math.ceil((float) N / Br);
			float scale = (float) (1.0f / Math.sqrt(d));
			
			int col_tile_size = Bc * d;  // size of Kj, Vj
		    int row_tile_size = Br * d;  // size of Qi
		    int sram_size = (4 * col_tile_size * Sizeof.FLOAT) + (3 * row_tile_size * Sizeof.FLOAT) + (2 * Bc * Br * Sizeof.FLOAT);
		  
	    	dQ.clearGPU();
	    	dK.clearGPU();
	    	dV.clearGPU();
		    
	        /**
	         *  设置入参
	         *  const float* Q,
			    const float* K,
			    const float* V,
			    const float* O,
			    const float* dO,
			    const float* L,
			    const int N,
			    const int d,
			    const int Tc,
			    const int Tr,
			    const int Bc,
			    const int Br,
			    const float softmax_scale,
			    float* dQ,
			    float* dK,
			    float* dV
	         */ 
			backwardKernelParameters = Pointer.to(
	        		Pointer.to(Q.getGpuData()),
	        		Pointer.to(K.getGpuData()),
	        		Pointer.to(V.getGpuData()),
	        		Pointer.to(output.getGpuData()),
	        		Pointer.to(delta.getGpuData()),
	        		Pointer.to(d_l.getGpuData()),
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(new int[]{d}),
	        		Pointer.to(new int[]{Tc}),
	        		Pointer.to(new int[]{Tr}),
	        		Pointer.to(new int[]{Bc}),
	        		Pointer.to(new int[]{Br}),
	        		Pointer.to(new float[]{scale}),
	        		Pointer.to(dQ.getGpuData()),
	        		Pointer.to(dK.getGpuData()),
	        		Pointer.to(dV.getGpuData())
	            );
	        
			int[] grid_dim = new int[] {B, nh, 1};
			int[] block_dim = new int[] {Br, 1, 1};
			
//			if(CUDAModules.props.sharedMemPerBlock < sram_size) {
//				System.err.printf("max share memory size:"+CUDAModules.props.sharedMemPerBlock + ".current size:"+sram_size);
//			}
			
		    checkCUDA(cuLaunchKernel(backward_function,
		    		grid_dim[0],  grid_dim[1], grid_dim[2],      // Grid dimension
		    		block_dim[0], block_dim[1], block_dim[2],      // Block dimension
		    		sram_size, null,               // Shared memory size and stream
		    		backwardKernelParameters, null // Kernel- and extra parameters
		        ));
//		    d_l.showDM(0);
//		    JCuda.cudaDeviceSynchronize();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public static void main(String[] args) {
		
		int batchSize = 16;
		int headNum = 8;
		int time = 512;
		int headDim = 64; //headDim
		int len = batchSize * headNum * time * headDim;
		
		Tensor Q = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
		Tensor K = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
		Tensor V = new Tensor(batchSize, headNum, time, headDim, RandomUtils.gaussianRandom(len, 0.1f), true);
		
		Tensor dQ = new Tensor(batchSize, headNum, time, headDim, true);
		Tensor dK = new Tensor(batchSize, headNum, time, headDim, true);
		Tensor dV = new Tensor(batchSize, headNum, time, headDim, true);
		
		Tensor output = new Tensor(batchSize, headNum, time, headDim, true);

		Tensor delta = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.1f, 0.1f), true);
		
		CUDAManager cudaManager = new CUDAManager(0);
		
		FlashAttentionV2Kernel kernel = new FlashAttentionV2Kernel(headNum, time, headDim, cudaManager);
		
		for(int i = 0;i<100;i++) {
			long startTime = System.nanoTime();
		    
			kernel.forward(Q, K, V, output);

			System.out.println((System.nanoTime() - startTime)/1e6+"ms.");
			
//			output.
			
			long startTime2 = System.nanoTime();
			
			kernel.backward(Q, K, V, output, delta, dQ, dK, dV);
			
			System.out.println((System.nanoTime() - startTime2)/1e6+"ms.");
			
//			dQ.showDMByOffset(0, 100);
//			dK.showDMByOffset(0, 100);
//			dV.showDMByOffset(0, 100);
			
//			dK.showDM();
//			dV.showDM();
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
		
		for(int Tr_i = 0;Tr_i<Tr;Tr_i++) {

			for(int tx = 0;tx<d;tx++) {
				if (tx < d) {
					for (int i = 0; i < Br; i++) {
				        Qi[i * d + tx] = Q[Tr_i * Br * d + i * d + tx];
				    }
			    }
			}
			
			System.out.println(JsonUtils.toJson(Qi));
			
		    for (int j = 0; j < Tc; j++){

				for(int tx = 0;tx<Bc;tx++) {
			        // Load Kj, Vj to SRAM
			        for (int x = 0; x < d; x++) {
			        	if(j * Bc + tx < N){
			        		Kj[(tx * d) + x] = K[(tile_size * j) + (tx * d) + x];
			        	}else{
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
	
	public static void mul_kA_BT(float[] A,float[] B,float[] C,int m,int n,int k,int d) {

		for(int tx = 0;tx<d;tx++) {
		    for (int y = 0; y < n; y++) {
		        float sum = 0;
		        for (int x = 0; x < k; x++) {
		            sum += A[(tx * k) + x] * B[(y * k) + x];
		        }
		        C[(n * tx) + y] = sum;
		    }
		} 
	}
	
}
