package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class UpSampleKernel2 extends BaseKernel{
	
	private int scale;
	
	private int ndim = 3;
	
	private int d1;
	
	private int d2;
	
	private int d3;
	
	private CUfunction forward_function;
	
	private CUfunction backward_function;
	
	private Pointer forwardKernelParameters;
	private Pointer backwardKernelParameters;
	
	public UpSampleKernel2(int scale,int ndim, CUDAManager cudaManager) {
		super(cudaManager);
		this.scale = scale;
		this.ndim = ndim;
		init();
	}
	
	public void initFunction() {
		
		try {

			if(forward_function == null) {
				
				forward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "upscale");
				
			}
			
			if(backward_function == null) {
				
				backward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "downscale");
				
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
	
	public void forward(Tensor input,Tensor output) {
		upsample(input, output);
	}
	
	public void backward(Tensor delta,Tensor diff) {
		upsampleDelta(delta, diff);
	}
	
	public void upsample(Tensor input,Tensor output) {

		try {
			
			if(ndim == 3) {
				d1 = output.number;
				d2 = output.channel;
				d3 = output.width;
			}else {
				d1 = output.channel;
				d2 = output.height;
				d3 = output.width;
			}
			
			this.N = input.number;
			
	        /**
	         * 设置入参
	         * const float *input, float *output, int no_elements,int scale_factor, int d1, int d2, int d3
	         */
			forwardKernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[]{output.dataLength}),
	                Pointer.to(new int[]{scale}),
	                Pointer.to(new int[]{d1}),
	                Pointer.to(new int[]{d2}),
	                Pointer.to(new int[]{d3})
	            );
	        
			int nthreads = 256;
			
			int n_xblocks = Math.min(Math.max((int)Math.ceil((float)output.dataLength / nthreads), 1), 65535);
			
			int n_yblocks = (int)Math.ceil((float)output.dataLength / (float)(n_xblocks * nthreads));
			
			int[] blocks = new int[] {n_xblocks, n_yblocks, 1};
			
			int[] threads = new int[] {nthreads, 1, 1};

			checkCUDA(cuLaunchKernel(forward_function,
					blocks[0],  blocks[1], blocks[2],      // Grid dimension
					threads[0], threads[1], threads[2],      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void upsampleDelta(Tensor delta,Tensor diff) {

		try {
			

			diff.clearGPU();
			
			if(ndim == 3) {
				d1 = diff.number;
				d2 = diff.channel;
				d3 = diff.width;
			}else {
				d1 = diff.channel;
				d2 = diff.height;
				d3 = diff.width;
			}
			
	        /**
	         * 设置入参
	         * float *gradInput_data, const float *gradOutput_data, int no_elements, int scale_factor, int d1, int d2, int d3
	         */
			backwardKernelParameters = Pointer.to(
					Pointer.to(diff.getGpuData()),
					Pointer.to(delta.getGpuData()),
					Pointer.to(new int[]{diff.dataLength}),
					Pointer.to(new int[]{scale}),
					Pointer.to(new int[]{d1}),
		            Pointer.to(new int[]{d2}),
		            Pointer.to(new int[]{d3})
	            );
			
			
			int nthreads = 256;

		    int n_xblocks = Math.min(Math.max((int)Math.ceil((float)diff.dataLength / nthreads), 1), 65535);
		    
		    int n_yblocks = (int)Math.ceil((float)diff.dataLength / (float)(n_xblocks * nthreads));
		    
			int[] blocks = new int[] {n_xblocks, n_yblocks, 1};
			
			int[] threads = new int[] {nthreads, 1, 1};
			
			checkCUDA(cuLaunchKernel(backward_function,
					blocks[0],  blocks[1], blocks[2],      // Grid dimension
					threads[0], threads[1], threads[2],      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        ));
			
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
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
    	
    	try {

        	int N = 2;
        	int C = 3;
        	int H = 4;
        	int W = 4;
        	int ndim = 4;
        	int scale = 2;

        	int oHeight = H * scale;
    		int oWidth = W * scale;
    		
        	float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        	
        	float[] d = RandomUtils.order(N * C * oHeight * oWidth, 0.1f, 0.1f);

        	Tensor input = new Tensor(N, C, H, W, x, true);
        	
        	Tensor output = new Tensor(N, C, oHeight, oWidth, true);
        	
        	Tensor delta = new Tensor(N, C, oHeight, oWidth, d, true);
        	delta.showShape();
        	Tensor diff = new Tensor(N, C, H, W, true);
        	
        	CUDAManager cudaManager = new CUDAManager(0);
        	
        	UpSampleKernel2 pooling = new UpSampleKernel2(scale, ndim, cudaManager);
        	
        	long start = System.nanoTime();

//        	for(int i = 0;i<2;i++) {
        		
            pooling.forward(input, output);
            
//        	}
        	
    		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");

        	input.showDM();
        	
        	output.showDM();

            pooling.backward(delta, diff);
            
            delta.showDM();
            
        	diff.showDM();
        	
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
    	
    }
    
}
