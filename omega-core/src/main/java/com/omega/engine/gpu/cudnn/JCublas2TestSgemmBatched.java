package com.omega.engine.gpu.cudnn;

import com.omega.common.tensor.Tensor;
import com.omega.utils.JsonUtils;
import com.omega.utils.RandomUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class JCublas2TestSgemmBatched {
    public static void main(String[] args) {
        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);
        testSgemmBatched(10, 5);
        int batch = 2;
        int m = 5;
        int n = 5;
        int o = 5;
        int k = 5;
        float[] a = RandomUtils.order(batch * m * n, 1, 0);
        float[] b = RandomUtils.order(batch * o * k, 1, 0);
        Tensor at = new Tensor(batch, 1, m, n, a, true);
        Tensor bt = new Tensor(batch, 1, o, k, b, true);
        Tensor ct = new Tensor(batch, 1, m, o, true);
    }

    public static boolean testSgemmBatched(int b, int n) {
        System.out.println("=== Testing Sgemm with " + b + " batches of size " + n + " ===");
        float alpha = 1.0f;
        float beta = 0.0f;
        int nn = n * n;
        // System.out.println("Creating input data...");
        float h_A[][] = new float[b][nn];
        float h_B[][] = new float[b][nn];
        float h_C[][] = new float[b][nn];
        float h_C_ref[][] = new float[b][nn];
        for (int i = 0; i < b; i++) {
            h_A[i] = JCublas2TestSgemmBatched.createRandomFloatData1D(nn);
            h_B[i] = JCublas2TestSgemmBatched.createRandomFloatData1D(nn);
            h_C[i] = JCublas2TestSgemmBatched.createRandomFloatData1D(nn);
            h_C_ref[i] = h_C[i].clone();
        }
        System.out.println("Performing Sgemm with Java...");
        //        sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);
        System.out.println("Performing Sgemm with JCublas2...");
        sgemmBatchesJCublas2(n, alpha, h_A, h_B, beta, h_C);
        // Print the test results
        boolean passed = true;
        for (int i = 0; i < b; i++) {
            passed &= JCublas2TestSgemmBatched.equalNorm1D(h_C[i], h_C_ref[i]);
        }
        System.out.println(String.format("testSgemm %s", passed ? "PASSED" : "FAILED"));
        return passed;
    }

    static void sgemmBatchesJCublas2(int n, float alpha, float h_A[][], float h_B[][], float beta, float h_C[][]) {
        //JCublas2.setLogLevel(LogLevel.LOG_DEBUGTRACE);
        int nn = n * n;
        int b = h_A.length;
        Pointer[] h_Aarray = new Pointer[b];
        Pointer[] h_Barray = new Pointer[b];
        Pointer[] h_Carray = new Pointer[b];
        for (int i = 0; i < b; i++) {
            h_Aarray[i] = new Pointer();
            h_Barray[i] = new Pointer();
            h_Carray[i] = new Pointer();
            JCuda.cudaMalloc(h_Aarray[i], nn * Sizeof.FLOAT);
            JCuda.cudaMalloc(h_Barray[i], nn * Sizeof.FLOAT);
            JCuda.cudaMalloc(h_Carray[i], nn * Sizeof.FLOAT);
            JCublas2.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(h_A[i]), 1, h_Aarray[i], 1);
            JCublas2.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(h_B[i]), 1, h_Barray[i], 1);
            JCublas2.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(h_C[i]), 1, h_Carray[i], 1);
        }
        Pointer d_Aarray = new Pointer();
        Pointer d_Barray = new Pointer();
        Pointer d_Carray = new Pointer();
        JCuda.cudaMalloc(d_Aarray, b * Sizeof.POINTER);
        JCuda.cudaMalloc(d_Barray, b * Sizeof.POINTER);
        JCuda.cudaMalloc(d_Carray, b * Sizeof.POINTER);
        JCuda.cudaMemcpy(d_Aarray, Pointer.to(h_Aarray), b * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(d_Barray, Pointer.to(h_Barray), b * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(d_Carray, Pointer.to(h_Carray), b * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
        cublasHandle handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        JCublas2.cublasSgemmBatched(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, n, n, n, Pointer.to(new float[]{alpha}), d_Aarray, n, d_Barray, n, Pointer.to(new float[]{beta}), d_Carray, n, b);
        for (int i = 0; i < b; i++) {
            JCublas2.cublasGetVector(nn, Sizeof.FLOAT, h_Carray[i], 1, Pointer.to(h_C[i]), 1);
            JCuda.cudaFree(h_Aarray[i]);
            JCuda.cudaFree(h_Barray[i]);
            JCuda.cudaFree(h_Carray[i]);
        }
        System.out.println(JsonUtils.toJson(h_C));
        JCuda.cudaFree(d_Aarray);
        JCuda.cudaFree(d_Barray);
        JCuda.cudaFree(d_Carray);
        JCublas2.cublasDestroy(handle);
    }

    static void sgemmJava(int n, float alpha, float A[][], float B[][], float beta, float C[][]) {
        for (int i = 0; i < A.length; i++) {
            sgemmJava(n, alpha, A, B, beta, C);
        }
    }

    static void sgemmJava(int n, float alpha, float A[], float B[], float beta, float C[]) {
        int i;
        int j;
        int k;
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                float prod = 0;
                for (k = 0; k < n; ++k) {
                    prod += A[k * n + i] * B[j * n + k];
                }
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }
        }
    }

    public static boolean equalNorm1D(float a[], float b[]) {
        return equalNorm1D(a, b, a.length);
    }

    public static boolean equalNorm1D(float a[], float b[], int n) {
        if (a.length < n || b.length < n) {
            return false;
        }
        float errorNorm = 0;
        float refNorm = 0;
        for (int i = 0; i < n; i++) {
            float diff = a[i] - b[i];
            errorNorm += diff * diff;
            refNorm += a[i] * a[i];
        }
        errorNorm = (float) Math.sqrt(errorNorm);
        refNorm = (float) Math.sqrt(refNorm);
        return (errorNorm / refNorm < 1e-6f);
    }

    public static float[] createRandomFloatData1D(int x) {
        float a[] = new float[x];
        for (int i = 0; i < x; i++) {
            a[i] = RandomUtils.randomFloat();
        }
        return a;
    }
}
