package com.omega.engine.nn.layer.lpips.gpu;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class LPIPSKernel extends BaseKernel {
    private CUfunction lpip_l2_function;
    private CUfunction lpip_l2_backward_function;
    private CUfunction scaling_function;
    private CUfunction scaling_backwad_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;

    public LPIPSKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 3;
        int H = 2;
        int W = 2;
        float[] data = RandomUtils.order(N * C * H * W, 1f, 1f);
        Tensor input = new Tensor(N, C, H, W, data, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor shift = new Tensor(1, 1, 1, 3, new float[]{-0.03f, -0.088f, -0.188f}, true);
        Tensor scale = new Tensor(1, 1, 1, 3, new float[]{0.458f, 0.448f, 0.45f}, true);
        CUDAManager cudaManager = new CUDAManager(0);
        LPIPSKernel kernel = new LPIPSKernel(cudaManager);
        kernel.scaling(input, shift, scale, output);
        output.showDM();
        kernel.scaling_backwad(input, scale, output);
        output.showDM();
    }

    public void initFunction() {
        try {
            if (lpip_l2_function == null) {
                lpip_l2_function = getCudaManager().getLocalFunctionByModule("lpipsKernel.cu", "lpip_l2_kernel");
            }
            if (lpip_l2_backward_function == null) {
                lpip_l2_backward_function = getCudaManager().getLocalFunctionByModule("lpipsKernel.cu", "lpip_l2_backward_kernel");
            }
            if (scaling_function == null) {
                scaling_function = getCudaManager().getLocalFunctionByModule("lpipsKernel.cu", "scaling_kernel");
            }
            if (scaling_backwad_function == null) {
                scaling_backwad_function = getCudaManager().getLocalFunctionByModule("lpipsKernel.cu", "scaling_back_kernel");
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
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void lpip_l2(Tensor x1, Tensor x2, Tensor output) {
        try {
            /**
             * 设置入参
             * float *x1,float *x2, float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{x1.getDataLength()}));
            cuLaunchKernel(lpip_l2_function, this.CAFFE_GET_BLOCKS(x1.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void lpip_l2_backward(Tensor delta, Tensor x1, Tensor x2, Tensor diff) {
        try {
            /**
             * 设置入参
             * float *delta,float *x1,float *x2, float *diff, int N

             */
        	Pointer forwardKernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{delta.getDataLength()}));
            cuLaunchKernel(lpip_l2_backward_function, this.CAFFE_GET_BLOCKS(delta.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scaling(Tensor x, Tensor shift, Tensor scale, Tensor output) {
        try {
            /**
             * 设置入参
             * int N,float *x,float *shift, float *scale, float *out, int C,int HW
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(shift.getGpuData()), Pointer.to(scale.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{x.getShape()[1]}), Pointer.to(new int[]{x.getShape()[2] * x.getShape()[3]}));
            cuLaunchKernel(scaling_function, this.CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scaling_backwad(Tensor dy, Tensor scale, Tensor dx) {
        try {
            /**
             * 设置入参
             * int N,float *dy, float *scale, float *dx, int C,int HW
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[]{dy.getDataLength()}), Pointer.to(dy.getGpuData()), Pointer.to(scale.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{dy.getShape()[1]}), Pointer.to(new int[]{dy.getShape()[2] * dy.getShape()[3]}));
            cuLaunchKernel(scaling_backwad_function, this.CAFFE_GET_BLOCKS(dy.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

