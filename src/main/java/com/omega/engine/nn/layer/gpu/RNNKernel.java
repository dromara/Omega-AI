package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

/**
 * RNN CUDA Kernel
 * @author Administrator
 *
 */
public class RNNKernel extends BaseKernel{
	
	private CUfunction bias_function;
	
	private CUfunction output_function;
	
	private CUfunction output_bias_function;
	
	private CUfunction back_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer biasKernelParameters;
	
	private Pointer outputKernelParameters;
	
	private Pointer outputBiasKernelParameters;
	
	private Pointer kernelBackParameters;
	
	public RNNKernel(CUDAManager cudaManager) {
		super(cudaManager);
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

			if(bias_function == null) {

				bias_function = getCudaManager().getLocalFunctionByModule("RNNKernel.cu", "add_bias");
				
			}
			
			if(output_function == null) {

				output_function = getCudaManager().getLocalFunctionByModule("RNNKernel.cu", "add_output");
				
			}
			
			if(output_bias_function == null) {

				output_bias_function = getCudaManager().getLocalFunctionByModule("RNNKernel.cu", "add_output_bias");
				
			}
			
			if(back_function == null) {

				back_function = getCudaManager().getLocalFunctionByModule("RNNKernel.cu", "backward_bias_conn_kernel");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void addBias(Tensor output,Tensor bias) {
		
		try {
			
			if(biasKernelParameters == null || output.number != this.N){

		        /**
		         * 设置入参
		         * float* output, float* biases, int batch, int n, int size
		         */ 
				biasKernelParameters = Pointer.to(
		        		Pointer.to(output.getGpuData()),
		                Pointer.to(bias.getGpuData()),
		                Pointer.to(new int[]{output.getNumber()}),
		                Pointer.to(new int[]{output.getWidth()}),
		                Pointer.to(new int[]{1})
		            );
		        
		        this.N = output.number;
		        
			}

			cuLaunchKernel(bias_function,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            biasKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void addBias(Tensor output,Tensor bias,int t) {
		
		try {
			
			if(biasKernelParameters == null || output.number != this.N){

		        /**
		         * 设置入参
		         * float* output, float* biases, int batch, int n, int size
		         */ 
				biasKernelParameters = Pointer.to(
		        		Pointer.to(output.getGpuData().withByteOffset(t * Sizeof.FLOAT)),
		                Pointer.to(bias.getGpuData()),
		                Pointer.to(new int[]{output.getNumber()}),
		                Pointer.to(new int[]{output.getWidth()}),
		                Pointer.to(new int[]{1})
		            );
		        
		        this.N = output.number;
		        
			}

			cuLaunchKernel(bias_function,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            biasKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void addOutputBias(Tensor o1,Tensor o2, Tensor bias, int t) {
		
		try {
			
			if(outputBiasKernelParameters == null || o1.number != this.N){

		        /**
		         * 设置入参
		         * float* output, float* biases, int batch, int n, int size
		         */ 
				outputBiasKernelParameters = Pointer.to(
		        		Pointer.to(o1.getGpuData().withByteOffset(t * Sizeof.FLOAT)),
		        		Pointer.to(o2.getGpuData().withByteOffset(t * Sizeof.FLOAT)),
		                Pointer.to(bias.getGpuData()),
		                Pointer.to(new int[]{o1.getNumber()}),
		                Pointer.to(new int[]{o1.getWidth()}),
		                Pointer.to(new int[]{1})
		            );
		        
		        this.N = o1.number;
		        
			}

			cuLaunchKernel(output_bias_function,
		            this.CAFFE_GET_BLOCKS(o1.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            outputBiasKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void addOutput(Tensor o1,Tensor o2,int t) {
		
		try {
			
			if(outputKernelParameters == null || o1.number != this.N){

		        /**
		         * 设置入参
		         * float* output, float* biases, int batch, int n, int size
		         */ 
				outputKernelParameters = Pointer.to(
		        		Pointer.to(o1.getGpuData().withByteOffset(t * Sizeof.FLOAT)),
		        		Pointer.to(o2.getGpuData().withByteOffset(t * Sizeof.FLOAT)),
		                Pointer.to(new int[]{o1.getNumber()}),
		                Pointer.to(new int[]{o1.getWidth()}),
		                Pointer.to(new int[]{1})
		            );
		        
		        this.N = o1.number;
		        
			}

			cuLaunchKernel(output_function,
		            this.CAFFE_GET_BLOCKS(o1.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            outputKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backwardBias(Tensor diffB,Tensor delta) {
		
		try {
			
			diffB.clearGPU();
			
			if(kernelBackParameters == null) {

		        /**
		         * 设置入参
		         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
		         */ 
				kernelBackParameters = Pointer.to(
		        		Pointer.to(diffB.getGpuData()),
		                Pointer.to(delta.getGpuData()),
		                Pointer.to(new int[]{delta.getNumber()}),
		                Pointer.to(new int[]{delta.getWidth()})
		            );
		        
			}
			
			cuLaunchKernel(back_function,
		            this.CAFFE_GET_BLOCKS(delta.getWidth()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelBackParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backwardBias(Pointer diffB,Pointer delta,int nunmber,int width) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			kernelBackParameters = Pointer.to(
	        		Pointer.to(diffB),
	                Pointer.to(delta),
	                Pointer.to(new int[]{nunmber}),
	                Pointer.to(new int[]{width})
	            );
		     
			cuLaunchKernel(back_function,
		            this.CAFFE_GET_BLOCKS(width),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelBackParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
}
