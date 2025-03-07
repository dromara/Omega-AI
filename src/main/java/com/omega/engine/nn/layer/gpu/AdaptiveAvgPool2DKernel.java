package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class AdaptiveAvgPool2DKernel extends BaseKernel{

	private CUfunction forward_function;
	private CUfunction backward_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	private Pointer backwardKernelParameters;
	
	public AdaptiveAvgPool2DKernel(CUDAManager cudaManager) {
		super(cudaManager);
		init();
	}
	
	public void initFunction() {
		
		try {

			if(forward_function == null) {
				
				forward_function = getCudaManager().getLocalFunctionByModule("AdaptiveAvgPool2DKernel.cu", "AdaptiveAvgPool2DKernel");
				
			}
			
			if(backward_function == null) {
				
				backward_function = getCudaManager().getLocalFunctionByModule("AdaptiveAvgPool2DKernel.cu", "AdaptiveAvgPool2DGradKernel");
				
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
	}
	
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor input,Tensor output) {
		
		try {

			if(input.number != this.N) {
				
				this.N = input.number;
				
		        /**
		         * 设置入参
		         * const uint size, const uint input_height, const uint input_width,
                   const uint output_height, const uint output_width, float *input_data,float *output_data
		         */
				forwardKernelParameters = Pointer.to(
		                Pointer.to(new int[]{input.number * input.channel}),
		                Pointer.to(new int[]{input.height}),
		                Pointer.to(new int[]{input.width}),
		                Pointer.to(new int[]{output.height}),
		                Pointer.to(new int[]{output.width}),
		                Pointer.to(input.getGpuData()),
		                Pointer.to(output.getGpuData())
		            );
		        
			}
			
			cuLaunchKernel(forward_function,
		            this.CAFFE_GET_BLOCKS(input.number * input.channel),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor delta,Tensor diff) {
		
		try {

			if(delta.number != this.N) {
				
				this.N = delta.number;
				
		        /**
		         * 设置入参
		         * const uint size, const uint input_height, const uint input_width,
                   const uint output_height, const uint output_width, float *input_data,float *output_data
		         */
				backwardKernelParameters = Pointer.to(
		                Pointer.to(new int[]{delta.number * diff.channel}),
		                Pointer.to(new int[]{delta.height}),
		                Pointer.to(new int[]{delta.width}),
		                Pointer.to(new int[]{diff.height}),
		                Pointer.to(new int[]{diff.width}),
		                Pointer.to(delta.getGpuData()),
		                Pointer.to(diff.getGpuData())
		            );
		        
			}
			
			cuLaunchKernel(backward_function,
		            this.CAFFE_GET_BLOCKS(delta.number * delta.channel),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
    public static void main(String args[]){	

    	int N = 2;
    	int C = 3;
    	int H = 4;
    	int W = 4;
    	int oHeight = 2;
		int oWidth = 2;
		
    	float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
    	
    	Tensor input = new Tensor(N, C, H, W, x, true);
    	
    	Tensor output = new Tensor(N, C, oHeight, oWidth, true);
    	
    	CUDAManager cudaManager = new CUDAManager(0);
    	
    	AdaptiveAvgPool2DKernel pooling = new AdaptiveAvgPool2DKernel(cudaManager);
    	
    	long start = System.nanoTime();

    	for(int i = 0;i<2;i++) {

        	pooling.forward(input, output);
        	
    	}
    	
		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");

    	input.showDM();
    	
    	output.showDM();
    	
    }

}
