package com.omega.engine.gpu;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;

public class ConvKernel extends CUDAKernel {
    private String id;
    private float[] x;
    private float[] kernel;
    private float[] out;
    private int C;
    private int H;
    private int W;
    private int ko;
    private int kh;
    private int kw;
    private int s;
    private int p;
    private int oHeight;
    private int oWidth;
    private int ih;
    private int iw;
    private int numKernels;
    private CUfunction function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private CUdeviceptr dx;
    private CUdeviceptr dy;
    private Pointer dA;
    private Pointer dC;
    private Pointer kernelParameters;

    public ConvKernel(String id, float[] out, int C, int H, int W, int ko, int kh, int kw, int s, int p, CUDAManager cudaManager) {
        super(cudaManager);
        this.id = id;
        this.C = C;
        this.H = H;
        this.W = W;
        this.ko = ko;
        this.kh = kh;
        this.kw = kw;
        this.s = s;
        this.p = p;
        this.oHeight = ((H + 2 * p - kh) / s) + 1;
        this.oWidth = ((W + 2 * p - kw) / s) + 1;
        this.out = out;
        this.ih = C * kh * kw;
        this.iw = oHeight * oWidth;
        this.numKernels = C * oHeight * oWidth;
        //		long start1 = System.nanoTime();
        init();
        //        System.out.println((System.nanoTime() - start1) / 1e6 + "ms.1111");
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("Im2colKernel.cu", "im2col_gpu_kernelV2");
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
        this.dx = CUDAMemoryManager.getDevice(C * H * W);
        if (kh == 1) {
            dy = dx;
        } else {
            this.dy = CUDAMemoryManager.getDevice(ih * iw);
        }
        this.dA = CUDAMemoryManager.getPointer(ko * ih);
        this.dC = CUDAMemoryManager.getPointer(ko * iw);
        //		this.dx = CUDAMemoryManager.getDevice(id + "_dx", C * H * W);
        //		this.dy = CUDAMemoryManager.getDevice(id + "_dy", ih * iw);
        //
        //        this.dA = CUDAMemoryManager.getPointer(id + "_dA", ko * ih);
        //        this.dC = CUDAMemoryManager.getPointer(id + "_dC", ko * iw);
        /**
         * 设置入参
         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

         */
        kernelParameters = Pointer.to(Pointer.to(dx), Pointer.to(dy), Pointer.to(new int[]{numKernels}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{kh}), Pointer.to(new int[]{kw}), Pointer.to(new int[]{s}), Pointer.to(new int[]{p}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}));
    }

    public void setX(float[] x) {
        this.x = x;
        /**
         * 申请内存

         */
        JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT);
    }

    public void setKernel(float[] kernel) {
        this.kernel = kernel;
        cublasSetVector(ko * ih, Sizeof.FLOAT, Pointer.to(kernel), 1, dA, 1);
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void conv() {
        //        long start = System.nanoTime();
        /**
         * if kernel size is 1 * 1.
         * im2col(input) = input

         */
        if (kh > 1) {
            im2col();
        }
        sgemm();
        //        System.out.println((System.nanoTime() - start) / 1e6 + "ms22222");
    }

    public void sgemm() {
        /**
         * m k n

         */
        getCudaManager().getOp().multiplyFloat(ko, iw, ih, dA, dy, dC, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
    }

    public void im2col() {
        try {
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(numKernels), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void free() {
        JCuda.cudaFree(dx);
        JCuda.cudaFree(dy);
        if (dA != null) {
            GPUOP.getInstance().free(dA);
            GPUOP.getInstance().free(dC);
        }
    }

    public float[] getOut() {
        cublasGetVector(out.length, Sizeof.FLOAT, dC, 1, Pointer.to(out), 1);
        return out;
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
    //    public static void main(String args[]){
    //    	int N = 2;
    //    	int C = 3;
    //    	int H = 8;
    //    	int W = 8;
    //    	int ko = 2;
    //    	int kh = 3;
    //    	int kw = 3;
    //    	int s = 1;
    //    	int p = 0;
    //    	int oHeight = ((H + 2 * p - kh) / s) + 1;
    //		int oWidth = ((W + 2 * p - kw) / s) + 1;
    //		int ow = oHeight * oWidth;
    //		int oh = ko;
    //
    ////    	float[] x1 = RandomUtils.gaussianRandom(N * C * H * W, 0.01f);
    ////
    ////    	float[] k1 = RandomUtils.gaussianRandom(ko * C * kh * kw, 0.01f);
    //
    //		float[] x1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
    //
    //    	float[] k1 = RandomUtils.order(ko * C * kh * kw, 0.1f, 0.1f);
    //
    //    	float[] out = new float[oh * ow];
    //
    //    	float[][][][] out2 = new float[N][ko][oHeight][oWidth];
    //
    //    	float[][][][] out3 = new float[N][ko][oHeight][oWidth];
    //
    //    	float[] once = new float[C * H * W];
    //
    //    	float[] allOut = new float[N * ko * oHeight * oWidth];
    //
    //		ConvKernel ck = new ConvKernel("conv1", out, C, H, W, ko, kh, kw, s, p);
    //
    //		ck.setKernel(k1);
    //
    //    	long start = System.nanoTime();
    //
    //		for(int c = 0;c<20;c++){
    //
    //	    	long start3 = System.nanoTime();
    //	    	for(int n = 0;n<N;n++) {
    ////	    		long start2 = System.nanoTime();
    //	    		System.arraycopy(x1, n * C * H * W, once, 0, C * H * W);
    //	    		ck.setX(once);
    //	        	ck.conv();
    //	        	System.arraycopy(ck.getOut(), 0, allOut, n * oh * ow, oh * ow);
    //	        	MatrixUtils.col2im4d(ck.getOut(), out2, n, ko, oHeight, oWidth);
    ////	        	System.out.println((System.nanoTime() - start2) / 1e6 + "ms.:"+i);
    //	    	}
    //
    //	    	System.out.println((System.nanoTime() - start3) / 1e6 + "ms================>c.:"+c);
    //
    //		}
    //
    //		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
    //
    //		System.out.println(JsonUtils.toJson(allOut));
    //
    //    	int ow2 = C * kh * kw;
    //		int oh2 = N * oHeight * oWidth;
    //
    //    	float[] im2col = new float[oh2 * ow2];
    //
    //    	float[][][][] x2 = MatrixUtils.transform(x1, N, C, H, W);
    //
    //    	float[][][][] k2 = MatrixUtils.transform(k1, ko, C, kh, kw);
    //
    //    	float[] ka = Im2colUtils.kernalToVector(k2, false);
    //
    //    	float[] kt = Im2colUtils.kernalToVector2(k2, false);
    //
    //    	System.out.println("k:"+CheckArrayUtils.check(k1, kt));
    //
    //    	float[] out1 = new float[N * oh * ow];
    //
    //    	long start2 = System.nanoTime();
    //
    //    	for(int i = 0;i<20;i++) {
    //    		long start1 = System.nanoTime();
    //
    //        	Im2colToVector.im2col(x2, im2col, kh, kw, s);
    //
    //        	float[] r = new float[N * oh * ow];
    //
    //        	int xm = N * oHeight * oWidth;
    //    		int xn = kh * kw * C;
    //
    //    		GPUOP.getInstance().multiplyFloat(xm, xn, ko, im2col, ka, r);
    //
    //	    	System.out.println((System.nanoTime() - start1) / 1e6 + "ms.cpu:"+i);
    //	    	out1 = r;
    //    	}
    //
    //    	System.out.println((System.nanoTime() - start2) / 1e6 + "ms.cpu-count");
    //
    //    	MatrixUtils.col2imgV2(out1, out3, N, ko, oHeight, oWidth);
    //
    //    	System.out.println(CheckArrayUtils.check(out2, out3));
    //
    //		CUDAMemoryManager.free();
    //    }
}

