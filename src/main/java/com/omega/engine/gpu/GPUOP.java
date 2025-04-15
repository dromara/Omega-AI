package com.omega.engine.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.*;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.jcurand.curandStatus;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import java.util.List;
import java.util.Locale;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.*;

public class GPUOP {
    private static GPUOP o;
    private cublasHandle handle;
    private curandGenerator generator;

    public GPUOP() {
        // TODO Auto-generated constructor stub
        this.handle = new cublasHandle();
        cublasCreate(getHandle());
    }

    public GPUOP(int deviceId) {
        // TODO Auto-generated constructor stub
        CUDAManager.checkCUDA(JCuda.cudaSetDevice(deviceId));
        this.handle = new cublasHandle();
        init();
        this.generator = getGenerator();
    }

    public static GPUOP getInstance() {
        if (GPUOP.o == null) {
            GPUOP.o = new GPUOP();
        }
        return GPUOP.o;
    }

    public static void main(String[] args) {
        //    	//1024,576,64
        //    	//256,576,128
        //    	//256,1152,128
        //    	//16,4608,512
        //        // 1024 * 27
        //    	//4:4608:512
        //
        //    	GPUOP.getInstance().init();
        //
        //    	int m = 256 * 128;
        //        int n = 1152;
        //        int k = 128;
        //
        //        float[] a = RandomUtils.gaussianRandom(m * n, 0.1f);
        //
        //        // 24 * 64
        //        float[] b = RandomUtils.gaussianRandom(n * k, 0.1f);
        //
        //        float[] c = new float [m * k];
        //
        //    	for(int i = 0;i<1280;i++) {
        //
        //	        long start = System.nanoTime();
        //
        //	        GPUOP.getInstance().multiplyFloat(m, n, k, a, b, c);
        //
        //	        System.out.println((System.nanoTime() - start)/1e6 + "ms");
        //
        ////	        System.out.println(JsonUtils.toJson(c));
        //
        //    	}
        //
        //    	GPUOP.getInstance().clear();
        //
        //    	System.out.println("=========================>");
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
        //    	test();
        //    	testBatch();
    }

    public static void test() {
        int m = 5;
        int n = 4;
        int o = 1;
        int k = 3;
        int time = n;
        float[] a = RandomUtils.order(m * n, 1, 1);
        float[] b = RandomUtils.order(o * k, 1, 1);
        Tensor at = new Tensor(m, 1, 1, n, a, true);
        Tensor bt = new Tensor(o, 1, 1, k, b, true);
        Tensor ct = new Tensor(m, 1, 1, k, true);
        for (int t = 0; t < time; t++) {
            GPUOP.getInstance().multiplyFloat(m, k, n, at.getGpuData().withByteOffset(t * Sizeof.FLOAT), bt.getGpuData(), ct.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
            PrintUtils.printImage(ct.syncHost());
            System.out.println("");
        }
    }

    public static void testBatch() {
        int batch = 2;
        int m = 5;
        int n = 3;
        int k = 4;
        float[] a = RandomUtils.order(batch * m * k, 1, 0);
        float[] b = RandomUtils.order(batch * n * k, 1, 0);
        Tensor at = new Tensor(batch, 1, m, k, a, true);
        Tensor bt = new Tensor(batch, 1, n, k, b, true);
        Tensor ct = new Tensor(batch, 1, m, n, true);
        //    	GPUOP.getInstance().launchMultiply(at.getGpuData(), bt.getGpuData(), ct.getGpuData(), batch, m, n, k, true, false, 0);
        GPUOP.getInstance().bmm(at.getGpuData(), bt.getGpuData(), ct.getGpuData(), batch, m, n, k, CUBLAS_OP_N, CUBLAS_OP_T);
        at.showDM();
        ct.showDM();
        System.out.println("");
    }

    public static String toString2D(float[] a, int columns) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            if ((i > 0) && (i % columns == 0)) {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }

    public static String toString2D(double[] a, int columns) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            if ((i > 0) && (i % columns == 0)) {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }

    public static int checkCUBLASResult(int result) {
        if (result != cublasStatus.CUBLAS_STATUS_SUCCESS) {
            System.err.println("cuda error code:" + result + "[" + cublasStatus.stringFor(result) + "]");
            throw new CudaException(cublasStatus.stringFor(result));
        }
        return result;
    }

    private static int checkCURANDResult(int result) {
        if (result != curandStatus.CURAND_STATUS_SUCCESS) {
            System.err.println("curand error code:" + result + "[" + curandStatus.stringFor(result) + "]");
            throw new CudaException(cublasStatus.stringFor(result));
        }
        return result;
    }

    public void init() {
    	checkCUBLASResult(cublasCreate(getHandle()));
    }

    public void clear() {
        cublasDestroy(getHandle());
    }

    public curandGenerator getGenerator() {
        if (generator == null) {
            generator = new curandGenerator();
            checkCURANDResult(JCurand.curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT));
        }
        return generator;
    }

    public void conv(float[] px, float[] ik, float[] out, int n, int c, int h, int w, int ko, int kh, int kw, int s) {
        Pointer dA = this.im2col(px, n, c, h, w, kh, kw, s);
        int oHeight = ((h - kh) / s) + 1;
        int oWidth = ((w - kw) / s) + 1;
        int xm = n * oHeight * oWidth;
        int xn = kh * kw * c;
        /**
         * 申请内存

         */
        //        Pointer dA = Pointer.to(di);
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        cudaMalloc(dB, ik.length * Sizeof.FLOAT);
        cudaMalloc(dC, out.length * Sizeof.FLOAT);
        cublasSetVector(ik.length, Sizeof.FLOAT, Pointer.to(ik), 1, dB, 1);
        //        cublasSetVector(out.length, Sizeof.FLOAT, Pointer.to(out), 1, dC, 1);
        Pointer zero = Pointer.to(new float[]{0.0f});
        Pointer one = Pointer.to(new float[]{1.0f});
        int status = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, ko, xm, xn, one, dB, ko, dA, xn, zero, dC, ko);
        //        cudaDeviceSynchronize();
        cublasGetVector(xm * ko, Sizeof.FLOAT, dC, 1, Pointer.to(out), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    private Pointer im2col(float[] px, int n, int c, int h, int w, int kh, int kw, int s) {
        CUfunction function = CUDAModules.getLocalFunctionByModule("Im2colKernel.cu", "im2col_gpuv4");
        int oH = ((h - kh) / s) + 1;
        int oW = ((w - kw) / s) + 1;
        int oh = n * oH * oW;
        int ow = c * kh * kw;
        int kSize = kh * kw;
        /**
         * 申请内存

         */
        CUdeviceptr deviceInputX = new CUdeviceptr();
        cuMemAlloc(deviceInputX, px.length * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoD(deviceInputX, Pointer.to(px), px.length * Sizeof.FLOAT);
        CUdeviceptr deviceInputOut = new CUdeviceptr();
        cuMemAlloc(deviceInputOut, oh * ow * Sizeof.FLOAT);
        Pointer dx = Pointer.to(deviceInputX);
        Pointer dy = Pointer.to(deviceInputOut);
        /**
         * 设置入参
         * int oHeight,int oWidth,int ow,int oh,int kSize

         */
        Pointer kernelParameters = Pointer.to(dx, dy, Pointer.to(new int[]{n}), Pointer.to(new int[]{c}), Pointer.to(new int[]{h}), Pointer.to(new int[]{w}), Pointer.to(new int[]{kh}), Pointer.to(new int[]{kw}), Pointer.to(new int[]{s}), Pointer.to(new int[]{oH}), Pointer.to(new int[]{oW}), Pointer.to(new int[]{oh}), Pointer.to(new int[]{ow}), Pointer.to(new int[]{kSize}));
        int blockSizeX = 1024;
        int gridSizeX = (oh * ow + blockSizeX - 1) / blockSizeX;
        if (oh * ow <= blockSizeX) {
            blockSizeX = oh * ow;
            gridSizeX = 1;
        }
        cuLaunchKernel(function, gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cudaDeviceSynchronize();
        // Clean up.
        cudaFree(dx);
        return dy;
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyFloat(int m, int n, int k, float A[], float B[], float C[]) {
        //    	long start = System.nanoTime();
        //    	cublasCreate(handle);
        //
        //
        //        System.out.println(m+":"+n+":"+k);
        //
        //
        try {
            Pointer dA = new Pointer();
            Pointer dB = new Pointer();
            Pointer dC = new Pointer();
            cudaMalloc(dA, m * n * Sizeof.FLOAT);
            cudaMalloc(dB, n * k * Sizeof.FLOAT);
            cudaMalloc(dC, m * k * Sizeof.FLOAT);
            cublasSetVector(m * n, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
            cublasSetVector(n * k, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
            //cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(C), 1, dC, 1);
            Pointer zero = Pointer.to(new float[]{0.0f});
            Pointer one = Pointer.to(new float[]{1.0f});
            int status = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, dB, k, dA, n, zero, dC, k);
            cudaDeviceSynchronize();
            cublasGetVector(m * k, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        // Clean up
        //        cublasDestroy(handle);
        //        System.out.println((System.nanoTime() - start) / 1e6+"ms。gpu");
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyFloat(int m, int n, int k, float[] C, Pointer dA, Pointer dB, Pointer dC) {
        //    	long start = System.nanoTime();
        //    	cublasCreate(handle);
        //
        //
        //        System.out.println(m+":"+n+":"+k);
        //
        //
        try {
            //    		Pointer dC = new Pointer();
            //            cudaMalloc(dA, m * n * Sizeof.FLOAT);
            //            cudaMalloc(dB, n * k * Sizeof.FLOAT);
            //              cudaMalloc(dC, m * k * Sizeof.FLOAT);
            //            cublasSetVector(m * n, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
            //            cublasSetVector(n * k, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
            //            cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(C), 1, dC, 1);
            Pointer zero = Pointer.to(new float[]{0.0f});
            Pointer one = Pointer.to(new float[]{1.0f});
            int status = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, dB, k, dA, n, zero, dC, k);
            //            cudaDeviceSynchronize();
            cublasGetVector(m * k, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
            //            cudaFree(dA);
            //            cudaFree(dB);
            //            cudaFree(dC);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void multiplyFloat(int m, int n, int k, float[] C, Pointer dA, Pointer dB, Pointer dC, int CUBLAS_OP_A, int CUBLAS_OP_N_B, float alpha, float beta) {
        try {
            Pointer zero = Pointer.to(new float[]{alpha});
            Pointer one = Pointer.to(new float[]{beta});
            int lda = CUBLAS_OP_A == CUBLAS_OP_N ? k : m;
            int ldb = CUBLAS_OP_N_B == CUBLAS_OP_N ? n : k;
            //            System.out.println(lda+":"+ldb);
            int status = cublasSgemm(getHandle(), CUBLAS_OP_N_B, CUBLAS_OP_A, n, m, k, one, dB, ldb, dA, lda, zero, dC, n);
            cublasGetVector(C.length, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void multiplyFloat(int m, int n, int k, Pointer dA, Pointer dB, Pointer dC, int CUBLAS_OP_A, int CUBLAS_OP_N_B, float alpha, float beta) {
        try {
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int lda = CUBLAS_OP_A == CUBLAS_OP_N ? k : m;
            int ldb = CUBLAS_OP_N_B == CUBLAS_OP_N ? n : k;
            int status = cublasSgemm(getHandle(), CUBLAS_OP_N_B, CUBLAS_OP_A, n, m, k, alphaP, dB, ldb, dA, lda, betaP, dC, n);
            checkCUBLASResult(status);
            //            cudaDeviceSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void multiplyFloatEX(int transa, int transb, int m, int n, int k, float alpha, /** host or device pointer */
                                Pointer A, int lda, Pointer B, int ldb, float beta, /** host or device pointer */
                                Pointer C, int ldc) {
        try {
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int status = JCublas2.cublasGemmEx(getHandle(), transb, transa, n, m, k, alphaP, B, cudaDataType.CUDA_R_32F, ldb, A, cudaDataType.CUDA_R_32F, lda, betaP, C, cudaDataType.CUDA_R_32F, ldc, cudaDataType.CUDA_R_32F, cublasGemmAlgo.CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            checkCUBLASResult(status);
            //            cudaDeviceSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void gemv(int trans_A, int M, int N, Tensor A, Tensor X, Tensor Y, float alpha, float beta) {
        try {
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int status = JCublas2.cublasSgemv(getHandle(), trans_A, N, M, alphaP, A.getGpuData(), N, X.getGpuData(), 1, betaP, Y.getGpuData(), 1);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void launchMultiply(Pointer a, Pointer b, Pointer c, int batch_size, int m, int n, int k, boolean trans_A, boolean trans_B, int scaler) {
        if (scaler == 0) {
            float alpha = 1.0f;
            float beta = 0.0f;
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            if (!trans_A && !trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alphaP, b, n, n * k, a, k, m * k, betaP, c, n, m * n, batch_size);
            } else if (!trans_A && trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alphaP, b, k, n * k, a, k, m * k, betaP, c, n, m * n, batch_size);
            } else if (trans_A && !trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alphaP, b, n, n * k, a, m, m * k, betaP, c, n, m * n, batch_size);
            } else if (trans_A && trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, alphaP, b, k, n * k, a, m, m * k, betaP, c, n, m * n, batch_size);
            }
        } else {
            float alpha = (float) Math.sqrt(1.0 / scaler);
            float beta = 0.0f;
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            if (!trans_A && !trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alphaP, b, n, n * k, a, k, m * k, betaP, c, n, m * n, batch_size);
            } else if (!trans_A && trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alphaP, b, k, n * k, a, k, m * k, betaP, c, n, m * n, batch_size);
            } else if (trans_A && !trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alphaP, b, n, n * k, a, m, m * k, betaP, c, n, m * n, batch_size);
            } else if (trans_A && trans_B) {
                JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, alphaP, b, k, n * k, a, m, m * k, betaP, c, n, m * n, batch_size);
            }
        }
    }

    public void bmm(Pointer dA, Pointer dB, Pointer dC, int batch_size, int m, int n, int k, int CUBLAS_OP_A, int CUBLAS_OP_B) {
        float alpha = 1.0f;
        float beta = 0.0f;
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int lda = CUBLAS_OP_A == CUBLAS_OP_N ? k : m;
        int ldb = CUBLAS_OP_B == CUBLAS_OP_N ? n : k;
        int status = JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_B, CUBLAS_OP_A, n, m, k, alphaP, dB, ldb, n * k, dA, lda, m * k, betaP, dC, n, m * n, batch_size);
        checkCUBLASResult(status);
    }

    public void bmm(Pointer dA, Pointer dB, Pointer dC, int batch_size, int m, int n, int k, int CUBLAS_OP_A, int CUBLAS_OP_B, float alpha, float beta) {
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int lda = CUBLAS_OP_A == CUBLAS_OP_N ? k : m;
        int ldb = CUBLAS_OP_B == CUBLAS_OP_N ? n : k;
        int status = JCublas2.cublasSgemmStridedBatched(getHandle(), CUBLAS_OP_B, CUBLAS_OP_A, n, m, k, alphaP, dB, ldb, n * k, dA, lda, m * k, betaP, dC, n, m * n, batch_size);
        checkCUBLASResult(status);
    }

    public void bmm(int transa, int transb, int m, int n, int k, float alpha, /** host or device pointer */
                    Pointer A, int lda, long strideA, /** purposely signed */
                    Pointer B, int ldb, long strideB, float beta, /** host or device pointer */
                    Pointer C, int ldc, long strideC, int batchCount) {
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int status = JCublas2.cublasSgemmStridedBatched(getHandle(), transa, transb, m, n, k, alphaP, A, lda, strideA, B, ldb, strideB, betaP, C, ldc, strideC, batchCount);
        checkCUBLASResult(status);
    }

    public void bmmEX(int transa, int transb, int m, int n, int k, float alpha, /** host or device pointer */
                      Pointer A, int lda, long strideA, /** purposely signed */
                      Pointer B, int ldb, long strideB, float beta, /** host or device pointer */
                      Pointer C, int ldc, long strideC, int batchCount) {
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int status = JCublas2.cublasGemmStridedBatchedEx(getHandle(), transa, transb, m, n, k, alphaP, A, cudaDataType.CUDA_R_32F, lda, strideA, B, cudaDataType.CUDA_R_32F, ldb, strideB, betaP, C, cudaDataType.CUDA_R_32F, ldc, strideC, batchCount, cudaDataType.CUDA_R_32F, cublasGemmAlgo.CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        checkCUBLASResult(status);
    }

    /**
     * CT = (A*B)T = BT * AT
     *
     * @param m
     * @param n
     * @param k
     * @param alpha
     * @param A
     * @param B
     * @param beta
     * @param C
     * @param batchCount
     */
    public void bmm(int m, int n, int k, float alpha, /** host or device pointer */
                    Tensor A, /** purposely signed */
                    Tensor B, float beta, /** host or device pointer */
                    Tensor C, int batchCount) {
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int transa = CUBLAS_OP_N;
        int transb = CUBLAS_OP_N;
        int lda = k;
        int ldb = n;
        int ldc = n;
        long strideA = m * k;
        long strideB = k * n;
        long strideC = m * n;
        int status = JCublas2.cublasGemmStridedBatchedEx(getHandle(), transa, transb, n, m, k, alphaP, B.getGpuData(), cudaDataType.CUDA_R_32F, ldb, strideB, A.getGpuData(), cudaDataType.CUDA_R_32F, lda, strideA, betaP, C.getGpuData(), cudaDataType.CUDA_R_32F, ldc, strideC, batchCount, cudaDataType.CUDA_R_32F, cublasGemmAlgo.CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        checkCUBLASResult(status);
    }

    public void bmm(int transa, int transb, int m, //A row
                    int n, //B col
                    int k, //A col and B row
                    float alpha, /** host or device pointer */
                    Tensor A, /** [m, k] */
                    Tensor B, /** [k, n] */
                    float beta, /** [m, n] */
                    Tensor C, int batchCount) {
        Pointer alphaP = Pointer.to(new float[]{alpha});
        Pointer betaP = Pointer.to(new float[]{beta});
        int lda = k; // transA = CUBLAS_OP_N
        if (transa == CUBLAS_OP_T) {
            lda = m;
        }
        int ldb = n; // transB = CUBLAS_OP_N
        int ldc = n; // transB = CUBLAS_OP_N
        if (transb == CUBLAS_OP_T) {
            ldb = k;
            ldc = k;
        }
        long strideA = m * k;
        long strideB = k * n;
        long strideC = m * n;
        int status = JCublas2.cublasGemmStridedBatchedEx(getHandle(), transb, transa, n, m, k, alphaP, B.getGpuData(), cudaDataType.CUDA_R_32F, ldb, strideB, A.getGpuData(), cudaDataType.CUDA_R_32F, lda, strideA, betaP, C.getGpuData(), cudaDataType.CUDA_R_32F, ldc, strideC, batchCount, cudaDataType.CUDA_R_32F, cublasGemmAlgo.CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        checkCUBLASResult(status);
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyFloat(int m, int n, int k, float A[], float B[], float C[], int CUBLAS_OP_A, int CUBLAS_OP_B, float alpha, float beta) {
        try {
            int lda = CUBLAS_OP_A == CUBLAS_OP_N ? k : m;
            int ldb = CUBLAS_OP_B == CUBLAS_OP_N ? n : k;
            Pointer dA = new Pointer();
            Pointer dB = new Pointer();
            Pointer dC = new Pointer();
            cudaMalloc(dA, m * k * Sizeof.FLOAT);
            cudaMalloc(dB, k * n * Sizeof.FLOAT);
            cudaMalloc(dC, m * n * Sizeof.FLOAT);
            cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
            cublasSetVector(k * n, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int status = cublasSgemm(getHandle(), CUBLAS_OP_B, CUBLAS_OP_A, n, m, k, alphaP, dB, ldb, dA, lda, betaP, dC, n);
            cudaDeviceSynchronize();
            cublasGetVector(m * n, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyFloat(int m, int n, int k, float A[], float B[], float C[], int CUBLAS_OP_A, int CUBLAS_OP_B, float alpha, float beta, int lda, int ldb, int ldc) {
        try {
            Pointer dA = new Pointer();
            Pointer dB = new Pointer();
            Pointer dC = new Pointer();
            cudaMalloc(dA, m * k * Sizeof.FLOAT);
            cudaMalloc(dB, k * n * Sizeof.FLOAT);
            cudaMalloc(dC, m * n * Sizeof.FLOAT);
            cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
            cublasSetVector(k * n, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int status = cublasSgemm(getHandle(), CUBLAS_OP_B, CUBLAS_OP_A, n, m, k, alphaP, dB, ldb, dA, lda, betaP, dC, ldc);
            cudaDeviceSynchronize();
            cublasGetVector(m * n, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void gpu_gemv(int m, int n, Pointer dA, Pointer dx, Pointer dy, int CUBLAS_OP_A, float alpha, float beta) {
        try {
            Pointer alphaP = Pointer.to(new float[]{alpha});
            Pointer betaP = Pointer.to(new float[]{beta});
            int status = JCublas2.cublasSgemv(getHandle(), CUBLAS_OP_A, n, m, alphaP, dA, n, dx, 1, betaP, dy, 1);
            cudaDeviceSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyFloatBatch(int m, int n, int k, int N, float[] A, List<float[]> B, List<float[]> C) {
        //    	long start = System.nanoTime();
        Pointer[] Apointers = new Pointer[N];
        Pointer[] Bpointers = new Pointer[N];
        Pointer[] Cpointers = new Pointer[N];
        for (int i = 0; i < N; ++i) {
            Apointers[i] = new Pointer();
            cudaMalloc(Apointers[i], m * n * Sizeof.FLOAT);
            JCuda.cudaMemcpy(Apointers[i], Pointer.to(A), m * n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Bpointers[i] = new Pointer();
            cudaMalloc(Bpointers[i], n * k * Sizeof.FLOAT);
            JCuda.cudaMemcpy(Bpointers[i], Pointer.to(B.get(i)), n * k * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Cpointers[i] = new Pointer();
            cudaMalloc(Cpointers[i], m * k * Sizeof.FLOAT);
        }
        Pointer d_Aarray = new Pointer();
        Pointer d_Barray = new Pointer();
        Pointer d_Carray = new Pointer();
        cudaMalloc(d_Aarray, m * n * Sizeof.POINTER);
        cudaMalloc(d_Barray, n * k * Sizeof.POINTER);
        cudaMalloc(d_Carray, m * k * Sizeof.POINTER);
        JCuda.cudaMemcpy(d_Aarray, Pointer.to(Apointers), m * n * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(d_Barray, Pointer.to(Bpointers), n * k * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(d_Carray, Pointer.to(Cpointers), m * k * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        Pointer zero = Pointer.to(new float[]{0.0f});
        Pointer one = Pointer.to(new float[]{1.0f});
        JCublas2.cublasSgemmBatched(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, d_Barray, k, d_Aarray, n, zero, d_Carray, k, N);
        cudaDeviceSynchronize();
        for (int i = 0; i < N; i++) {
            JCuda.cudaMemcpy(Pointer.to(C.get(i)), Cpointers[i], m * k * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cudaFree(Apointers[i]);
            cudaFree(Bpointers[i]);
            cudaFree(Cpointers[i]);
        }
        //        System.out.println((System.nanoTime() - start) / 1e6+"ms。gpu");
    }

    public void cudaRandom(Tensor x) {
        try {
            checkCURANDResult(JCurand.curandSetPseudoRandomGeneratorSeed(getGenerator(), RandomUtils.rand()));
            checkCURANDResult(JCurand.curandGenerateUniform(getGenerator(), x.getGpuData(), x.getDataLength()));
            //    		System.out.println("in");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cudaRandn(Tensor x) {
        try {
            checkCURANDResult(JCurand.curandSetPseudoRandomGeneratorSeed(getGenerator(), RandomUtils.rand()));
            checkCURANDResult(JCurand.curandGenerateNormal(getGenerator(), x.getGpuData(), x.getDataLength(), 0, 1));
            //    		System.out.println("in");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void free(Pointer p) {
        cudaFree(p);
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     *
     * @param size The size of the matrices
     * @param A    Matrix A
     * @param B    Matrix B
     * @param C    Matrix C
     */
    public void multiplyDouble(int m, int n, int k, double A[], double B[], double C[]) {
        cublasCreate(getHandle());
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        cudaMalloc(dA, m * n * Sizeof.DOUBLE);
        cudaMalloc(dB, n * k * Sizeof.DOUBLE);
        cudaMalloc(dC, m * k * Sizeof.DOUBLE);
        cublasSetVector(m * n, Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
        cublasSetVector(n * k, Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
        cublasSetVector(m * k, Sizeof.DOUBLE, Pointer.to(C), 1, dC, 1);
        Pointer zero = Pointer.to(new float[]{0.0f});
        Pointer one = Pointer.to(new float[]{1.0f});
        int status = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, dB, k, dA, n, zero, dC, k);
        cudaDeviceSynchronize();
        cublasGetVector(m * k, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        // Clean up
        cublasDestroy(getHandle());
    }

    public cublasHandle getHandle() {
        return handle;
    }
}

