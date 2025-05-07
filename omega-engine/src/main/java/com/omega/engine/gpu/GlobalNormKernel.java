package com.omega.engine.gpu;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class GlobalNormKernel extends CUDAKernel {
    private int CAFFE_CUDA_NUM_THREADS = 512;
    private CUfunction globalNorm_gpu_function;
    private CUfunction globalNorm_gpu_function2;

    public GlobalNormKernel(CUDAManager cudaManager) {
        super(cudaManager);
        globalNorm_gpu_function = cudaManager.getLocalFunctionByModule("GlobalNormKernel.cu", "vectorL2NormKernel");
        globalNorm_gpu_function2 = cudaManager.getLocalFunctionByModule("GlobalNormKernel.cu", "l2NormKernel");
    }

    public void globalNorm(Tensor output, Tensor input) {
        try {
            output.clearGPU();
            /**
             * const float* a, float* result, int n

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.dataLength}));
            checkCUDA(cuLaunchKernel(globalNorm_gpu_function, CAFFE_GET_BLOCKS(input.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void globalNorm2(Tensor output, Tensor input) {
        try {
            output.clearGPU();
            /**
             * const float* a, float* result, int n

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.dataLength}));
            checkCUDA(cuLaunchKernel(globalNorm_gpu_function2, CAFFE_GET_BLOCKS(input.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void showDM(Pointer p, float[] data) {
        JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        System.out.println(JsonUtils.toJson(data));
    }

    public void showDM(Pointer p, int[] data) {
        JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        System.out.println(JsonUtils.toJson(data));
    }

    public void showDM(Pointer p, int size) {
        float[] data = new float[size];
        JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        System.out.println(JsonUtils.toJson(data));
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
    //	public static void main(String[] args) {
    //
    //		try {
    //
    //			CUDAModules.initContext();
    //
    //			int N = 1024;
    //			int W = 1024;
    //
    //			float[] x_data = MatrixUtils.order(N * W, 1f, 1f);
    //
    //			float tmp = 0.0f;
    //			for(int i = 0;i<x_data.length;i++){
    //				tmp += x_data[i] * x_data[i];
    //			}
    //
    //			System.out.println(Math.sqrt(tmp));
    //
    //			Tensor x = new Tensor(1, 1, N, W, x_data, true);
    //
    //			GlobalNormKernel kernel = new GlobalNormKernel();
    //
    //			Tensor output = new Tensor(1, 1, 1, 1, true);
    //
    //			Tensor output2 = new Tensor(1, 1, 1, 1, true);
    //
    //			Tensor y = new Tensor(1, 1, 1, 1, true);
    //
    //			kernel.globalNorm(output, x);
    //
    ////			kernel.globalNorm2(output2, x);
    ////
    ////			NormalizeKernel kernel2 = new NormalizeKernel();
    ////
    ////			kernel2.norm(x, y);
    //
    //
    //			System.out.println(Math.sqrt(output.syncHost()[0]));
    //
    //			System.out.println(Math.sqrt(output2.syncHost()[0]));
    //
    //			System.out.println(y.syncHost()[0]);
    //
    ////			output.showDM();
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
}

