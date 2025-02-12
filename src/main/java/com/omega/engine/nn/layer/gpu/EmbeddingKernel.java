package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

public class EmbeddingKernel extends BaseKernel{

	private CUfunction function;
	
	private CUfunction function2;
	
	private CUfunction back_function;
	
	private CUfunction get_time_embedding_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private int BLOCK = 512;
	
	private Pointer kernelParameters;
	
	private Pointer kernelBackParameters;
	
	public EmbeddingKernel() {
		init();
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();

	}
	
	public void initFunction() {
		
		try {

			if(function == null) {

				function = CUDAModules.getLocalFunctionByModule("EmbeddingKernel.cu", "EmbeddingFW");
				
			}
			
			if(function2 == null) {

				function2 = CUDAModules.getLocalFunctionByModule("EmbeddingKernel.cu", "embedding_forward_kernel");
				
			}
			
			if(get_time_embedding_function == null) {

				get_time_embedding_function = CUDAModules.getLocalFunctionByModule("EmbeddingKernel.cu", "get_time_embedding");
				
			}
			
			if(back_function == null) {

				back_function = CUDAModules.getLocalFunctionByModule("EmbeddingKernel.cu", "EmbeddingGrad");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public int get_number_of_blocks(int array_size, int block_size)
	{
		return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
	}
	
	public void forward(Tensor input,Tensor weight,Tensor output) {
		
		try {

//			input.showDMByOffsetRed(0, 76, "input:");
//			input.showDM();
	        /**
	         * 设置入参
	         *  float *output,
                const float *table,
                const float *ids,
                const int N,
                const int K,
                const int D
	         */ 
			Pointer kernelParameters = Pointer.to(
		        		Pointer.to(output.getGpuData()),
		        		Pointer.to(weight.getGpuData()),
		        		Pointer.to(input.getGpuData()),
		        		Pointer.to(new int[]{weight.height}),
		        		Pointer.to(new int[]{input.getDataLength()}),
		        		Pointer.to(new int[]{weight.width})
		            );
//			weight.showShape("weight");
		    this.N = input.number;

			int gridx = 2 * CUDAModules.props.multiProcessorCount;

		    int[] threads = new int[] {256, 4, 1};
		    int[] grids = new int[] {gridx, 1, 1};
			
		    checkCUDA(cuLaunchKernel(function,
					grids[0], grids[1], grids[2],      // Grid dimension
					threads[0], threads[1], threads[2],
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

	       
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward2(Tensor input,Tensor weight,Tensor output) {
		
		try {

//			input.showDMByOffsetRed(0, 76, "input:");
//			input.showDM();
	        /**
	         * 设置入参
	         *  const uint32_t n_elements,
			    const uint32_t stride,
			    const uint32_t n_dim,
			    const float* __restrict__ params,
			    const float* __restrict__ indices,
			    float* __restrict__ output
	         */ 
			Pointer kernelParameters = Pointer.to(
						Pointer.to(new int[]{input.getDataLength()}),
		        		Pointer.to(new int[]{weight.height}),
		        		Pointer.to(new int[]{weight.width}),
		        		Pointer.to(weight.getGpuData()),
		        		Pointer.to(input.getGpuData()),
		        		Pointer.to(output.getGpuData())
		            );
//			weight.showShape("weight");
		    this.N = input.number;

		    checkCUDA(cuLaunchKernel(function2,
		    		this.CAFFE_GET_BLOCKS(input.getDataLength()), 1, 1,      // Grid dimension
		    		CAFFE_CUDA_NUM_THREADS, 1, 1,
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

	       
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor delta,Tensor dw,Tensor input) {
		
		try {
			
//			dw.valueGPU(0);
			
			if(kernelBackParameters == null || delta.number != this.N){

		        /**
		         * 设置入参
		         * float* table,
                   const float* output,
                   const float* ids,
                   const int N,
                   const int K,
                   const int D
		         */ 
				kernelBackParameters = Pointer.to(
		        		Pointer.to(dw.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(input.getGpuData()),
		                Pointer.to(new int[]{dw.height}),
		        		Pointer.to(new int[]{input.dataLength}),
		        		Pointer.to(new int[]{dw.width})
		            );
		        
		        this.N = delta.number;
		        
			}
			
			int gridx = 2 * CUDAModules.props.multiProcessorCount;
		    int[] threads = new int[] {128, 8, 1};
		    int[] grids = new int[] {gridx, 1, 1};

		    checkCUDA(cuLaunchKernel(back_function,
					grids[0],  grids[1], grids[2],      // Grid dimension
					threads[0], threads[1], threads[2],
		            0, null,               // Shared memory size and stream
		            kernelBackParameters, null // Kernel- and extra parameters
		        ));
//			delta.showDMByNumber(0);
//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void get_time_embedding(Tensor input,Tensor factor,Tensor output,int dim) {
		
		try {

	        /**
	         * 设置入参
	         *  float* input, float* factor, float* output, int N,int dim
	         */ 
	        kernelParameters = Pointer.to(
	        		Pointer.to(input.getGpuData()),
	        		Pointer.to(factor.getGpuData()),
	        		Pointer.to(output.getGpuData()),
	        		Pointer.to(new int[]{input.number * dim}),
	        		Pointer.to(new int[]{dim})
	            );
	        
	        this.N = input.number;

		    checkCUDA(cuLaunchKernel(get_time_embedding_function,
		    		this.get_number_of_blocks(input.getDataLength(), BLOCK),  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

//	        JCudaDriver.cuCtxSynchronize();
//	        output.syncHost();
//	        output.showDMByNumber(0);
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
		
		String[] txts = new String[] {
				"sharp focus on the cats eyes.",
				"cinematic bokeh: ironcat, sharp focus on the cat's eyes",
				"sharp focus on the cats eyes.",
				"cinematic bokeh: ironcat, sharp focus on the cat's eyes"
		};
		
		Tensor label = new Tensor(77, 1, 1, 1, true);
		
		String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
		String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
		BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
		
		Tensor output = new Tensor(77, 1, 1, 512, true);

		Tensor weight = new Tensor(1, 1, 49408, 512, MatrixUtils.order(49408 * 512, 0.00001f, 0.0001f), true);
		
		for(int i = 0;i<4;i++) {

			String txt = txts[i];
			System.err.println(txt);
			int[] ids = bpe.encodeInt(txt, 77);
			
			for(int j = 0;j<77;j++) {
				if(j<ids.length) {
					label.data[j] = ids[j];
				}else {
					label.data[j] = 0;
				}
			}
			
			label.hostToDevice();
			label.showDM();
			
			EmbeddingKernel kernel = new EmbeddingKernel();
	    	
	    	kernel.forward(label, weight, output);
	    	
	    	output.showDMByOffsetRed(512, 1024, "in");
		}
    	
	}
	
}
