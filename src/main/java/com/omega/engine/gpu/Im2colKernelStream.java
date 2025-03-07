package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class Im2colKernelStream extends CUDAKernel{

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

	
	public Im2colKernelStream(float[] x,float[] out,int N,int C,int H,int W,int kh,int kw,int s,CUDAManager cudaManager) {
		super(cudaManager);
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
		this.oh = N * oHeight * oWidth;
		this.ow = C * kh * kw;
		this.kSize = kh * kw;
		this.out = out;
		initFunction();
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				function = getCudaManager().getLocalFunctionByModule("Im2colKernelTmp.cu", "im2col_gpuv4");
//				
//				/**
//				 * 加载方法
//				 */
//				CUmodule module = CUDAModules.getModule(LibPaths.LIB_PATH+"Im2colKernel.cu");
//				
//		        // Obtain a function pointer to the "add" function.
//				function = new CUfunction();
//		        cuModuleGetFunction(function, module, "im2col_gpuv4");
//		        
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void im2col() {
		
		try {
//			long start1 = System.nanoTime();
			
			CUstream stream = new CUstream();
			JCudaDriver.cuStreamCreate(stream, 1);
			
			/**
			 * 申请内存
			 */
	        CUdeviceptr deviceInputX = new CUdeviceptr();
	        cuMemAlloc(deviceInputX, x.length * Sizeof.FLOAT);
//	        cuMemcpyHtoD(deviceInputX, Pointer.to(x), x.length * Sizeof.FLOAT);
	        
	        CUdeviceptr deviceInputOut = new CUdeviceptr();
	        cuMemAlloc(deviceInputOut, out.length * Sizeof.FLOAT);
	        
//	        long start3 = System.nanoTime();
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
	                Pointer.to(new int[]{oh}),
	                Pointer.to(new int[]{ow}),
	                Pointer.to(new int[]{kSize})
	            );
//	        System.out.println((System.nanoTime() - start3) / 1e6 + "ms3");
	        // Call the kernel function.
	        int blockSizeX = 1024;
//	        int numBlocks = (N + blockSizeX - 1) / blockSizeX;
//	        int gridSizeX = (int)Math.ceil((double)oh * ow / blockSizeX);
	        int gridSizeX = (oh * ow + blockSizeX - 1) / blockSizeX;
//	        int gridSizeY = (ow + CUDAModules.threadsPerDimension - 1) / CUDAModules.threadsPerDimension;
//	        System.out.println(oh);
//	        System.out.println(blockSizeX);
	        if(oh * ow <= blockSizeX) {
	        	blockSizeX = oh * ow;
	        	gridSizeX = 1;
	        }
//	       System.out.println(gridSizeX+":"+gridSizeY);
//	        System.out.println(gridSizeX+":"+blockSizeX);
	        
//	        long start2 = System.nanoTime();
	        
	        for(int i = 0;i<1;i++) {
	        	
	        	JCudaDriver.cuMemcpyHtoDAsync(deviceInputX, Pointer.to(x), x.length * Sizeof.FLOAT, stream);
	        	
		        cuLaunchKernel(function,
		            gridSizeX,  1, 1,      // Grid dimension
		            blockSizeX, 1, 1,      // Block dimension
		            0, stream,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        
		        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(out), deviceInputOut, out.length * Sizeof.FLOAT, stream);
		        
	        }
	        
	        JCudaDriver.cuStreamSynchronize(stream);
	        
//	        cuCtxSynchronize();
//	        System.out.println((System.nanoTime() - start2) / 1e6 + "ms2");
	        
//	        long start4 = System.nanoTime();
//	        System.out.println(out.length);
//	        cuMemcpyDtoH(Pointer.to(out), deviceInputOut, out.length * Sizeof.FLOAT);

	        // Clean up.
	        JCuda.cudaFree(deviceInputX);
	        JCuda.cudaFree(deviceInputOut);
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

    public static void main(String args[]){	
//
//    	CUmodule module = CUDAModules.getModule(LibPaths.LIB_PATH+"Im2colKernel.cu");
//		
//        // Obtain a function pointer to the "add" function.
//    	CUfunction function = new CUfunction();
//    	
//        cuModuleGetFunction(function, module, "im2col_gpuv4");
//        int N = 64;
//        int C = 3;
//        int H = 224;
//        int W = 224;
//        int S = 1;
//        int kSize = 7;
//        
//        int ow = (W + 0 - kSize) / S + 1;
//		int oh = (H + 0 - kSize) / S + 1;
//        int pLength = C * kSize * kSize * N * oh * ow;
//        int n = N * oh * ow;
//        int m = C * kSize * kSize;
//        
//        System.out.println("pLength:"+pLength);
//        
//        float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
//        
//        float[] o = new float[pLength];
//        
//        long start = System.nanoTime();
//        
//        CUdeviceptr deviceX = new CUdeviceptr();
//        cuMemAlloc(deviceX, x.length * Sizeof.FLOAT);
//        
//        CUdeviceptr deviceO = new CUdeviceptr();
//        cuMemAlloc(deviceO, o.length * Sizeof.FLOAT);
//        
//        cuMemcpyHtoD(deviceX, Pointer.to(x),
//                x.length * Sizeof.FLOAT);
//        
//        /**
//         * 设置入参
//         * float *x,float *out,int N,int C,int H,int W,int kh,int kw,int stride,int oHeight,int oWidth,int ow,int oh,int kSize
//         */
//        Pointer kernelParameters = Pointer.to(
//        		Pointer.to(deviceX),
//        		Pointer.to(deviceO),
//                Pointer.to(new int[]{N}),
//                Pointer.to(new int[]{C}),
//                Pointer.to(new int[]{H}),
//                Pointer.to(new int[]{W}),
//                Pointer.to(new int[]{kSize}),
//                Pointer.to(new int[]{kSize}),
//                Pointer.to(new int[]{S}),
//                Pointer.to(new int[]{oh}),
//                Pointer.to(new int[]{ow}),
//                Pointer.to(new int[]{n}),
//                Pointer.to(new int[]{m}),
//                Pointer.to(new int[]{kSize * kSize})
//            );
//        System.out.println((System.nanoTime() - start) / 1e6 + "ms");
//        
//        System.out.println("n:"+n);
//        
//        // Call the kernel function.
//        int blockSizeX = 1024;
////        int blockSizeY = 1;
//        int gridSizeX = (n * m + blockSizeX - 1) / blockSizeX;
////        int gridSizeY = (m + blockSizeY - 1) / blockSizeY;
//        
//        
//        long start2 = System.nanoTime();
//        cuLaunchKernel(function,
//            gridSizeX,  1, 1,      // Grid dimension
//            blockSizeX, 1, 1,      // Block dimension
//            0, null,               // Shared memory size and stream
//            kernelParameters, null // Kernel- and extra parameters
//        );
//        cuCtxSynchronize();
//        
//        cuMemcpyDtoH(Pointer.to(o), deviceO, o.length * Sizeof.FLOAT);
//        System.out.println((System.nanoTime() - start2) / 1e6 + "ms");
//        System.out.println(o[o.length - 1]);
//        
//        JCuda.cudaFree(deviceX);
//        JCuda.cudaFree(deviceO);
        
//    	int N = 128;
//    	int C = 64;
//    	int H = 64;
//    	int W = 64;
//    	int kh = 3;
//    	int kw = 3;
//    	int s = 1;
//    	int oHeight = ((H - kh ) / s) + 1;
//		int oWidth = ((W - kw) / s) + 1;
//		int ow = C * kh * kw;
//		int oh = N * oHeight * oWidth;
//    	
//    	float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
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
//        	Im2colKernelStream k = new Im2colKernelStream(x, out, N, C, H, W, kh, kw, s);
//        	
//        	k.im2col();
//        	
//        	System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
//        	
//    	}
//    	
////    	System.out.println(JsonUtils.toJson(out));
//    	
//    	System.out.println("==============================>");
//		
//    	float[] out2 = new float[oh * ow];
//    	
//	    for(int i = 0;i<10;i++) {
//	    	
//	    	long start2 = System.nanoTime();
//	//    	
//	//    	float[] cpu = MatrixOperation.im2col4d(x, N, C, H, W, kh, kw, s);
//	//    	
//	    	Im2colToVector.im2col(x2, out2, kh, kw, s);
//	 
//	//		
//	    	System.out.println((System.nanoTime() - start2) / 1e6 + "ms");
//    	}
////    	System.out.println(JsonUtils.toJson(cpu));
//    	
//	    System.out.println(CheckArrayUtils.check(out, out2));
	    
    }

	public float[] getOut() {
		return out;
	}

}
