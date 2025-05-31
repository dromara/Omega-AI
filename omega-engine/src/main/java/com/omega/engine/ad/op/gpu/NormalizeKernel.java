package com.omega.engine.ad.op.gpu;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import java.io.Serializable;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class NormalizeKernel extends BaseKernel implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 3345793649705471080L;
    public int N = 0;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private CUfunction norm_function;
    private CUfunction l2_norm_function;
    private CUfunction l2_norm_backward_function;
    private CUfunction l2_norm_1dim_function;
    private CUfunction l2_norm_1dim_backward_function;
    private CUfunction l2_norm_1dim_backward_function2;
    private CUfunction l2_norm_1dim_backward_function3;

    //	private CUfunction norm_grad_function;
    public NormalizeKernel(CUDAManager cudaManager) {
        super(cudaManager);
        norm_function = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "norm");
        l2_norm_function = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_kernel");
        l2_norm_backward_function = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_backward_kernel");
        l2_norm_1dim_function = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_1dim_kernel");
        l2_norm_1dim_backward_function = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_1dim_backward_kernel");
        l2_norm_1dim_backward_function2 = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_1dim_backward_kernel2");
        l2_norm_1dim_backward_function3 = cudaManager.getLocalFunctionByModule("NormalizeKernel.cu", "l2norm_1dim_backward_kernel3");
        //		norm_grad_function = CUDAModules.getLocalFunctionByModule("NormalizeKernel.cu", "NormalizeGradientKernel");
    }

    public static void main(String[] args) {
        int N = 2;
        int C = 3;
        int H = 2;
        int W = 2;
        float[] data = RandomUtils.order(N * C * H * W, 0.0001f, 0.0001f);
        Tensor input = new Tensor(N, C, H, W, data, true);
        Tensor dx = new Tensor(N, C, H, W, true);
        Tensor output = new Tensor(N, C, H, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        NormalizeKernel kernel = new NormalizeKernel(cudaManager);
        //    	kernel.l2norm1Dim(input, output);
        //
        //    	input.showDM();
        //
        //    	output.showDM();
        //
        float[] data2 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        //    	float[] data2 = MatrixUtils.one(N * C * H * W);
        Tensor delta = new Tensor(N, C, H, W, data2, true);
        //
        ////    	kernel.l2norm1Dim_back(input, output, delta, dx);
        //
        //    	kernel.l2norm1Dim_back2(input, delta, dx);
        //
        //    	dx.showDM();
//        input.view(N * C * H, 1, 1, W);
//        output.view(N * C * H, 1, 1, W);
        kernel.l2norm1Dim(input, output);
        input.showDM();
        output.showDM();
//        input.view(N * C * H, 1, 1, W);
//        output.view(N * C * H, 1, 1, W);
//        delta.view(N * C * H, 1, 1, W);
//        dx.view(N * C * H, 1, 1, W);
        kernel.l2norm1Dim_back3(input, delta, dx);
        dx.showDM();

        //    	Tensor output2 = new Tensor(N, C, 1, 1, true);
        //
        //    	TensorOP.mean2Dim(input, output2);
        //    	output2.showDM();
        //
        //    	delta.showDM();
        //
        //    	kernel.l2norm_back(input, output, delta, dx);
        //
        //    	dx.showDM();
        //    	Tensor input2 = new Tensor(N, 1, 1, W, data, true);
        //
        //    	Tensor output2 = new Tensor(1, 1, 1, 1, true);
        //    	long start = System.nanoTime();
        //    	for(int i = 0;i<10;i++) {
        ////        	System.out.println("output:");
        //        	kernel.norm(input, output);
        ////        	output.showDM();
        //    	}
        //    	output.showDM();
        //    	System.out.println((System.nanoTime() - start)/1e6+"ms.");
        //    	long start2 = System.nanoTime();
        //    	for(int i = 0;i<10;i++) {
        //    		output2.valueGPU(0);
        //    		TensorOP.pow(input, 2, input2);
        //        	TensorOP.sum(input2, output2, 0);
        //        	TensorOP.sqrt(output2, output2);
        //    	}
        //    	output2.showDM();
        //    	System.out.println((System.nanoTime() - start2)/1e6+"ms.");
    }

    public void norm(Tensor x, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(norm_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void l2norm(Tensor x, Tensor y) {
        try {
            /**
             * int N, float *x,float *out, float *dx, int filters

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.channel}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.height * x.width}));
            checkCUDA(cuLaunchKernel(l2_norm_function, CAFFE_GET_BLOCKS(x.number * x.channel), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void l2norm_back(Tensor x, Tensor out, Tensor delta, Tensor dx) {
        try {
            /**
             * int N, float *x,float *out,float *delta, float *dx, int filters

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.channel}), Pointer.to(x.getGpuData()), Pointer.to(out.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{x.height * x.width}));
            checkCUDA(cuLaunchKernel(l2_norm_backward_function, CAFFE_GET_BLOCKS(x.number * x.channel), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void l2norm1Dim(Tensor x, Tensor y) {
        try {
            /**
             * int N, float *x,float *out, int batch, int filters, int spatial, float eps

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.height * x.width}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.channel}), Pointer.to(new int[]{x.height * x.width}), Pointer.to(new float[]{1e-10f}));
            checkCUDA(cuLaunchKernel(l2_norm_1dim_function, CAFFE_GET_BLOCKS(x.number * x.height * x.width), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void l2norm1Dim_back(Tensor x, Tensor out, Tensor delta, Tensor dx) {
        try {
            /**
             * int N, float *x,float *out,float *delta, float *dx, int batch, int filters, int spatial, float eps

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.height * x.width}), Pointer.to(x.getGpuData()), Pointer.to(out.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.channel}), Pointer.to(new int[]{x.height * x.width}), Pointer.to(new float[]{1e-10f}));
            checkCUDA(cuLaunchKernel(l2_norm_1dim_backward_function, CAFFE_GET_BLOCKS(x.number * x.height * x.width), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void l2norm1Dim_back2(Tensor x, Tensor delta, Tensor dx) {
        try {
            /**
             * int N, float *x,float *delta, float *dx, int batch, int filters, int spatial, float eps

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.height * x.width}), Pointer.to(x.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.channel}), Pointer.to(new int[]{x.height * x.width}), Pointer.to(new float[]{1e-10f}));
            checkCUDA(cuLaunchKernel(l2_norm_1dim_backward_function2, CAFFE_GET_BLOCKS(x.number * x.height * x.width), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void l2norm1Dim_back3(Tensor x, Tensor delta, Tensor dx) {
        try {
            /**
             * int N, float *x,float *delta, float *dx, int batch, int filters, int spatial, float eps

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.height * x.width}), Pointer.to(x.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.channel}), Pointer.to(new int[]{x.height * x.width}), Pointer.to(new float[]{1e-10f}));
            checkCUDA(cuLaunchKernel(l2_norm_1dim_backward_function3, CAFFE_GET_BLOCKS(x.number * x.height * x.width), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
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

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

