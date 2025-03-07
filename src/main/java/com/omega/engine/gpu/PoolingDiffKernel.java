package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.pooling.PoolingType;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

public class PoolingDiffKernel extends CUDAKernel{
	private PoolingType type;
	private float[] x;
	private float[] out;
	private float[] mask;
	private int C;
	private int H;
	private int W;
	private int ph;
	private int pw;
	private int s;
	private int oHeight;
	private int oWidth;
	private int numKernels;
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;

	private CUdeviceptr dx;
	private CUdeviceptr dy;
	private CUdeviceptr dm;
	
	private Pointer kernelParameters;
	
	public PoolingDiffKernel(PoolingType type,float[] out,int C,int H,int W,int ph,int pw,int s,CUDAManager cudaManager) {
		super(cudaManager);
		this.type = type;
		this.C = C;
		this.H = H;
		this.W = W;
		this.ph = ph;
		this.pw = pw;
		this.s = s;
		this.oHeight = (H - ph) / s + 1;
		this.oWidth = (W - pw) / s + 1;
		this.numKernels = C * oHeight * oWidth;
		this.out = out;
		init();
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				switch (type) {
				case MAX_POOLING:

					function = getCudaManager().getLocalFunctionByModule("PoolingKernel.cu", "pooling_diff");

					break;
				case MEAN_POOLING:

					function = getCudaManager().getLocalFunctionByModule("PoolingKernel.cu", "pooling_diff");

					break;
				}
				
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

		/**
		 * 申请显存
		 */
		this.dx = CUDAMemoryManager.getDevice(C * oHeight * oWidth);
		this.dm = CUDAMemoryManager.getDevice(C * oHeight * oWidth * ph * pw);
		this.dy = CUDAMemoryManager.getDevice(C * H * W);
		
        /**
         * 设置入参
         * float* x,float* mask,float* result,int n,int height,int width,int oHeight,int oWidth,int pWidth,int pHeight,int stride
         */
        kernelParameters = Pointer.to(
        		Pointer.to(dx),
        		Pointer.to(dm),
                Pointer.to(dy),
                Pointer.to(new int[]{numKernels}),
                Pointer.to(new int[]{H}),
                Pointer.to(new int[]{W}),
                Pointer.to(new int[]{oHeight}),
                Pointer.to(new int[]{oWidth}),
                Pointer.to(new int[]{ph}),
                Pointer.to(new int[]{pw}),
                Pointer.to(new int[]{s})
            );
        
	}
	
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void diff() {
		
		try {
//			long start1 = System.nanoTime();
			
			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(numKernels),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
	        
//	        cuCtxSynchronize();

	        JCudaDriver.cuMemcpyDtoH(Pointer.to(out), dy, out.length * Sizeof.FLOAT);
	        
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void setX(float[] x) {
		this.x = x;
		JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT);
	}
	
	public void setMask(float[] mask) {
		this.mask = mask;
		JCudaDriver.cuMemcpyHtoD(dm, Pointer.to(mask), mask.length * Sizeof.FLOAT);
	}
	
	public float[] getOut() {
		return out;
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public void free() {
		JCudaDriver.cuMemFree(dx);
		JCudaDriver.cuMemFree(dm);
		JCudaDriver.cuMemFree(dy);
	}

//    public static void main(String args[]){	
//
//    	int N = 2;
//    	int C = 3;
//    	int H = 8;
//    	int W = 8;
//    	int ph = 2;
//    	int pw = 2;
//    	int s = 2;
//    	int oHeight = (H - ph) / s + 1;
//		int oWidth = (W - pw) / s + 1;
//
//    	float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
//    	
//    	float[] delta = RandomUtils.order(N * C * oHeight * oWidth, 0.1f, 0.1f);
//    	
//    	float[] once = new float[C * H * W];
//    	
//    	float[] onceDelta = new float[C * oHeight * oWidth];
//    	
//    	float[] out = new float[C * oHeight * oWidth];
//    	
//    	float[] mask = new float[C * oHeight * oWidth * ph * pw];
//    	
//    	float[] diff = new float[C * H * W];
//    	
//    	PoolingKernel pooling = new PoolingKernel(PoolingType.MAX_POOLING, out, mask, C, H, W, ph, pw, s);
//    	
//    	PoolingDiffKernel poolingDiff = new PoolingDiffKernel(PoolingType.MAX_POOLING, diff, C, H, W, ph, pw, s);
//    	
//    	long start = System.nanoTime();
//    	
////		for(int c = 0;c<20;c++){
////
////	    	long start3 = System.nanoTime();
//	    	for(int n = 0;n<N;n++) {
//	    		System.arraycopy(x, n * C * H * W, once, 0, C * H * W);
//	    		pooling.setX(once);
//	        	pooling.pooling();
//	        	System.arraycopy(delta, n * C * oHeight * oWidth, onceDelta, 0, C * oHeight * oWidth);
//	    		poolingDiff.setX(onceDelta);
//	    		poolingDiff.setMask(pooling.getMask());
//	    		poolingDiff.diff();
//	    		System.out.println(JsonUtils.toJson(diff));
//	    	}
////	    	System.out.println((System.nanoTime() - start3) / 1e6 + "ms================>c.:"+c);
////	    	
////		}
//		
//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
////    	System.out.println(JsonUtils.toJson(out));
////    	System.out.println(JsonUtils.toJson(mask));
////    	System.out.println(JsonUtils.toJson(diff));
//    	
//    	pooling.free();
//		poolingDiff.free();
//	    
////	    System.out.println(JsonUtils.toJson(out));
////	    
////	    System.out.println(JsonUtils.toJson(x));
////	    
////	    System.out.println(JsonUtils.toJson(xout));
//	    
//    }

}
