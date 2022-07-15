package com.omega.engine.gpu;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.JCublas2.cublasSgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
//import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.Locale;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;

public class GPUOP {
    
	private static GPUOP o;
	
	private cublasHandle handle;
	
	public GPUOP() {
		// TODO Auto-generated constructor stub
		this.handle = new cublasHandle();
		cublasCreate(handle);
	}
	
	public void init() {
		cublasCreate(handle);
	}
	
	public void clear() {
		cublasDestroy(handle);
	}
	
	/**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    public void multiplyFloat(int m,int n,int k, float A[],
        float B[], float C[])
    {
    	
//    	long start = System.nanoTime();
//    	cublasCreate(handle);

        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, m * n * Sizeof.FLOAT);
        cudaMalloc(dB, n * k * Sizeof.FLOAT);
        cudaMalloc(dC, m * k * Sizeof.FLOAT);
        cublasSetVector(m * n, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(n * k, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
        cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(C), 1, dC, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });

        int status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, 
            dB, k, dA, n, zero, dC, k);

        cudaDeviceSynchronize();
        
        cublasGetVector(m * k, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
       
        // Clean up
//        cublasDestroy(handle);

//        System.out.println((System.nanoTime() - start) / 1e6+"ms。gpu");
         
    }
    
	/**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    public void multiplyDouble( int m,int n,int k, double A[],
			double B[], double C[])
    {
    	cublasCreate(handle);
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, m * n * Sizeof.DOUBLE);
        cudaMalloc(dB, n * k * Sizeof.DOUBLE);
        cudaMalloc(dC, m * k * Sizeof.DOUBLE);
        cublasSetVector(m * n, Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
        cublasSetVector(n * k, Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
        cublasSetVector(m * k, Sizeof.DOUBLE, Pointer.to(C), 1, dC, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        int status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, 
            dB, k, dA, n, zero, dC, k);

        cudaDeviceSynchronize();
        
        cublasGetVector(m * k, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        // Clean up
        cublasDestroy(handle);
    }
    
    public static GPUOP getInstance() {
    	
    	if(GPUOP.o == null) {
    		GPUOP.o = new GPUOP();
    	}
    	
    	return GPUOP.o;
    }
    
    public static void main(String[] args){
    	
    	//1024,576,64
    	//256,576,128
    	//256,1152,128
    	//16,4608,512
        // 1024 * 27
    	//4:4608:512
    	
    	GPUOP.getInstance().init();
    	
    	int m = 256 * 128;
        int n = 1152;
        int k = 128;
        
        float[] a = RandomUtils.gaussianRandom(m * n, 0.1f);
        
        // 24 * 64
        float[] b = RandomUtils.gaussianRandom(n * k, 0.1f);

        float[] c = new float [m * k];
        
    	for(int i = 0;i<30;i++) {

	        long start = System.nanoTime();
	        
	        GPUOP.getInstance().multiplyFloat(m, n, k, a, b, c);

	        System.out.println((System.nanoTime() - start)/1e6 + "ms");
    	    
//	        System.out.println(JsonUtils.toJson(c));
	        
    	}
    	
    	GPUOP.getInstance().clear();

    	System.out.println("=========================>");
    	
//    	for(int i = 0;i<2;i++) {
//
//        	float[][] a = RandomUtils.gaussianRandom(4 * 128, 4608, 0.1f);
//        	
//        	float[][] b = RandomUtils.gaussianRandom(4608, 512, 0.1f);
//
//    		long start = System.nanoTime();
//    		
//        	MatrixOperation.multiplicationByEjml(a, b);
//        	
//        	System.out.println((System.nanoTime() - start)/1e6 + "ms");
//    	}

        
    }
    
    public static String toString2D(float[] a,int columns){
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
    public static String toString2D(double[] a,int columns){
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
}
