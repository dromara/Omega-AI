package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class Col2imKernel extends CUDAKernel{
	
	private float[] x;
	private float[] out;
	private int N;
	private int C;
	private int H;
	private int W;
	private int kh;
	private int kw;
	private int s;
	private int p;
	private int oHeight;
	private int oWidth;
	private int ow;
	private int oh;
	private int numKernels;
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;

	
	public Col2imKernel(float[] x,float[] out,int N,int C,int H,int W,int kh,int kw,int s,int p,CUDAManager cudaManager) {
		super(cudaManager);
		this.x = x;
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.p = p;
		this.oHeight = ((H + 2 * p - kh ) / s) + 1;
		this.oWidth = ((W + 2 * p - kw) / s) + 1;
		this.oh = N * oHeight * oWidth;
		this.ow = C * kh * kw;
		this.numKernels = C * H * W;
		this.out = out;
		initFunction();
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				function = getCudaManager().getLocalFunctionByModule("Col2imKernel.cu", "col2im_gpu_kernelV2");

			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void col2im() {
		
		try {
//			long start1 = System.nanoTime();

			/**
			 * 申请内存
			 */
	        CUdeviceptr dx = new CUdeviceptr();
	        cuMemAlloc(dx, x.length * Sizeof.FLOAT);

	        JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT);
	        
	        CUdeviceptr dy = new CUdeviceptr();
	        cuMemAlloc(dy, out.length * Sizeof.FLOAT);
	        
//	        long start3 = System.nanoTime();
	        /**
	         * 设置入参
	         * int oHeight,int oWidth,int ow,int oh,int kSize
	         */
	        Pointer kernelParameters = Pointer.to(
	        		Pointer.to(dx),
	                Pointer.to(dy),
	                Pointer.to(new int[]{numKernels}),
	                Pointer.to(new int[]{H}),
	                Pointer.to(new int[]{W}),
	                Pointer.to(new int[]{C}),
	                Pointer.to(new int[]{kh}),
	                Pointer.to(new int[]{kw}),
	                Pointer.to(new int[]{p}),
	                Pointer.to(new int[]{s}),
	                Pointer.to(new int[]{oHeight}),
	                Pointer.to(new int[]{oWidth})
	            );
	        
	        cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(numKernels),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
	        
//	        cuCtxSynchronize();

	        JCudaDriver.cuMemcpyDtoH(Pointer.to(out), dy, out.length * Sizeof.FLOAT);
	        
	        // Clean up.
	        JCuda.cudaFree(dx);
	        JCuda.cudaFree(dy);
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	public float[] getOut() {
		return out;
	}

//    public static void main(String args[]){	
//
//    	int N = 1;
//    	int C = 1;
//    	int H = 8;
//    	int W = 8;
//    	int kh = 3;
//    	int kw = 3;
//    	int s = 1;
//    	int p = 1;
//    	int oHeight = ((H + 2 * p - kh) / s) + 1;
//		int oWidth = ((W + 2 * p - kw) / s) + 1;
//		int oh = C * kh * kw;
//		int ow = oHeight * oWidth;
//    	
//    	float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
//    	
//    	float[] out = new float[oh * ow];
//    	
//    	float[] xout = new float[C * H * W];
//    	
//	    Im2colKernel im2col = new Im2colKernel(x, out, N, C, H, W, kh, kw, s, p);
//    	
//	    Col2imKernel col2im = new Col2imKernel(out, xout, N, C, H, H, kh, kw, s, p);
//	    
//	    im2col.im2col();
//	    
//	    col2im.col2im();
//	    
//	    col2im.col2im();
//	    
//	    System.out.println(x.length+":"+xout.length);
//	    
//	    System.out.println(JsonUtils.toJson(out));
//	    
//	    System.out.println(JsonUtils.toJson(x));
//	    
//	    System.out.println(JsonUtils.toJson(xout));
//	    
//    }

}
