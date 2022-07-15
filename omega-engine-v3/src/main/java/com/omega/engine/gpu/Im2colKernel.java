package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import com.omega.common.utils.Im2col4d;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

public class Im2colKernel {

	private float[] x;
	private float[] out;
	private int N;
	private int C;
	private int H;
	private int W;
	private int kh;
	private int kw;
	private int s;
	private int oHeight;
	private int oWidth;
	private int ow;
	private int oh;
	private int kSize;
	private CUfunction function;

	
	public Im2colKernel(float[] x,float[] out,int N,int C,int H,int W,int kh,int kw,int s) {
		this.x = x;
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.oHeight = ((H - kh ) / s) + 1;
		this.oWidth = ((W - kw) / s) + 1;
		this.ow = C * kh * kw;
		this.oh = N * oHeight * oWidth;
		this.kSize = kh * kw;
		this.out = out;
		initFunction();
	}
	
	public void initFunction() {
		
		try {

			/**
			 * 加载方法
			 */
			CUmodule module = CUDAModules.getModule("H://Im2colKernel.cu");
			
	        // Obtain a function pointer to the "add" function.
			function = new CUfunction();
	        cuModuleGetFunction(function, module, "im2col_gpu");
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public float[] im2col() {
		
		try {

			/**
			 * 申请内存
			 */
	        CUdeviceptr deviceInputX = new CUdeviceptr();
	        cuMemAlloc(deviceInputX, x.length * Sizeof.FLOAT);
	        cuMemcpyHtoD(deviceInputX, Pointer.to(x), x.length * Sizeof.FLOAT);
	        
	        CUdeviceptr deviceInputOut = new CUdeviceptr();
	        cuMemAlloc(deviceInputOut, out.length * Sizeof.FLOAT);
	        
	        /**
	         * 设置入参
	         * int oHeight,int oWidth,int ow,int oh,int kSize
	         */
	        Pointer kernelParameters = Pointer.to(
	                Pointer.to(deviceInputX),
	                Pointer.to(deviceInputOut),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{C}),
	                Pointer.to(new int[]{H}),
	                Pointer.to(new int[]{W}),
	                Pointer.to(new int[]{kh}),
	                Pointer.to(new int[]{kw}),
	                Pointer.to(new int[]{s}),
	                Pointer.to(new int[]{oHeight}),
	                Pointer.to(new int[]{oWidth}),
	                Pointer.to(new int[]{ow}),
	                Pointer.to(new int[]{oh}),
	                Pointer.to(new int[]{kSize})
	            );
	       
	        // Call the kernel function.
	        int blockSizeX = 256;
//	        int numBlocks = (N + blockSizeX - 1) / blockSizeX;
	        int gridSizeX = (int)Math.ceil((double)oh / blockSizeX);
//	        System.out.println(oh);
	        if(oh <= 256) {
	        	blockSizeX = oh;
	        	gridSizeX = 1;
	        }
	       
	        long start2 = System.nanoTime();
	        cuLaunchKernel(function,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        System.out.println((System.nanoTime() - start2) / 1e6 + "ms2");
//	        System.out.println((System.nanoTime() - start2) / 1e6 + "ms2");
	        
//	        long start1 = System.nanoTime();
//	        System.out.println(out.length);
	        cuMemcpyDtoH(Pointer.to(out), deviceInputOut, out.length * Sizeof.FLOAT);
	        
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
	        // Clean up.
	        cuMemFree(deviceInputX);
	        cuMemFree(deviceInputOut);
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return out;
	}

    public static void main(String args[]){	
    	
    	CUmodule module = CUDAModules.getModule("H://Im2colKernel.cu");
		
        // Obtain a function pointer to the "add" function.
    	CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test");
        int n = 100000;
        int m = 100;
        int[] o = new int[n * m];
        
        CUdeviceptr deviceInputOut = new CUdeviceptr();
        cuMemAlloc(deviceInputOut, o.length * Sizeof.INT);
        
        /**
         * 设置入参
         * int oHeight,int oWidth,int ow,int oh,int kSize
         */
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(deviceInputOut)
            );
       
        // Call the kernel function.
        int blockSizeX = 512;
        int blockSizeY = 1;
        int gridSizeX = (n + blockSizeX - 1) / blockSizeX;
        int gridSizeY = (n + blockSizeY - 1) / blockSizeY;
        
        System.out.println(gridSizeX);
        long start2 = System.nanoTime();
        cuLaunchKernel(function,
            gridSizeX,  gridSizeY, 1,      // Grid dimension
            blockSizeX, blockSizeY, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        cuMemcpyDtoH(Pointer.to(o), deviceInputOut, o.length * Sizeof.INT);
        System.out.println((System.nanoTime() - start2) / 1e6 + "ms");
        System.out.println(o[o.length-1]);
        
//    	int N = 128;
//    	int C = 512;
//    	int H = 32;
//    	int W = 32;
//    	int kh = 3;
//    	int kw = 3;
//    	int s = 1;
//    	int oHeight = ((H - kh ) / s) + 1;
//		int oWidth = ((W - kw) / s) + 1;
//		int ow = C * kh * kw;
//		int oh = N * oHeight * oWidth;
//    	
//    	float[] x = RandomUtils.gaussianRandom(N * C * H * W, 1);
//    	float[][][][] x2 = MatrixUtils.transform(x, N, C, H, W);
//
//    	float[] out = new float[oh * ow];
//    	
////    	System.out.println(x.length+"start.");
//
//    	for(int i = 0;i<10;i++) {
//
//    		long start = System.nanoTime();
//    		
//        	Im2colKernel k = new Im2colKernel(x, out, N, C, H, W, kh, kw, s);
//        	
//        	out = k.im2col();
//        	
//        	System.out.println((System.nanoTime() - start) / 1e6 + "ms");
//        	
//    	}
//    	
////    	System.out.println(JsonUtils.toJson(out));
//    	
//    	System.out.println("==============================>");
//		
//    	float[][] out2 = new float[oh][ow];
//    	
//	    for(int i = 0;i<10;i++) {
//	    	
//	    	long start2 = System.nanoTime();
//	//    	
//	//    	float[] cpu = MatrixOperation.im2col4d(x, N, C, H, W, kh, kw, s);
//	//    	
//	    	out2 = Im2col4d.im2colV2(x2, out2, kh, kw, s);
//	//		
//	    	System.out.println((System.nanoTime() - start2) / 1e6 + "ms");
//    	}
////    	System.out.println(JsonUtils.toJson(cpu));
    	
    }

	public float[] getOut() {
		return out;
	}

}
