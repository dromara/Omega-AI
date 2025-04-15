package com.omega.engine.loss.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class HingeLossKernel extends BaseKernel {
    private CUfunction hinge_d_loss_function;
    private CUfunction hinge_d_loss_back_function;
    private CUfunction hinge_d_real_loss_function;
    private CUfunction hinge_d_fake_loss_function;
    private CUfunction hinge_d_loss_real_back_function;
    private CUfunction hinge_d_loss_fake_back_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;

    public HingeLossKernel(CUDAManager cudaManager) {
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
        HingeLossKernel kernel = new HingeLossKernel(cudaManager);
    }

    public void initFunction() {
        try {
            if (hinge_d_loss_function == null) {
                hinge_d_loss_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_loss_kernel");
            }
            if (hinge_d_loss_back_function == null) {
                hinge_d_loss_back_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_loss_back_kernel");
            }
            if (hinge_d_real_loss_function == null) {
                hinge_d_real_loss_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_real_loss_kernel");
            }
            if (hinge_d_fake_loss_function == null) {
                hinge_d_fake_loss_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_fake_loss_kernel");
            }
            if (hinge_d_loss_real_back_function == null) {
                hinge_d_loss_real_back_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_loss_real_back_kernel");
            }
            if (hinge_d_loss_fake_back_function == null) {
                hinge_d_loss_fake_back_function = getCudaManager().getLocalFunctionByModule("HingeLossKernel.cu", "hinge_d_loss_fake_back_kernel");
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

    public void hingeLoss(Tensor real, Tensor fake, Tensor output) {
        try {
            /**
             * 设置入参
             * float *real, float *fake,float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(real.getGpuData()), Pointer.to(fake.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{real.getDataLength()}));
            cuLaunchKernel(hinge_d_loss_function, this.CAFFE_GET_BLOCKS(real.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void hingeRealLoss(Tensor real, Tensor output) {
        try {
            /**
             * 设置入参
             * float *real,float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(real.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{real.getDataLength()}));
            cuLaunchKernel(hinge_d_real_loss_function, this.CAFFE_GET_BLOCKS(real.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void hingeFakeLoss(Tensor fake, Tensor output) {
        try {
            /**
             * 设置入参
             * float *fake,float *out, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(fake.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{fake.getDataLength()}));
            cuLaunchKernel(hinge_d_fake_loss_function, this.CAFFE_GET_BLOCKS(fake.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void hingeLossBackward(Tensor real, Tensor fake, Tensor dReal, Tensor dFake) {
        try {
            /**
             * 设置入参
             * float *real, float *fake,float *dreal,float * dfake, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(real.getGpuData()), Pointer.to(fake.getGpuData()), Pointer.to(dReal.getGpuData()), Pointer.to(dFake.getGpuData()), Pointer.to(new int[]{real.getDataLength()}));
            cuLaunchKernel(hinge_d_loss_back_function, this.CAFFE_GET_BLOCKS(real.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void hingeLossRealBackward(Tensor real, Tensor dReal, float weight) {
        try {
            /**
             * 设置入参
             * float *real,float *dreal,float weight, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(real.getGpuData()), Pointer.to(dReal.getGpuData()), Pointer.to(new float[]{weight}), Pointer.to(new int[]{real.getDataLength()}));
            cuLaunchKernel(hinge_d_loss_real_back_function, this.CAFFE_GET_BLOCKS(real.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void hingeLossFakeBackward(Tensor fake, Tensor dFake, float weight) {
        try {
            /**
             * 设置入参
             * float *fake,float * dfake, float weight, int N

             */
            forwardKernelParameters = Pointer.to(Pointer.to(fake.getGpuData()), Pointer.to(dFake.getGpuData()), Pointer.to(new float[]{weight}), Pointer.to(new int[]{fake.getDataLength()}));
            cuLaunchKernel(hinge_d_loss_fake_back_function, this.CAFFE_GET_BLOCKS(fake.getDataLength()), 1, 1,      // Grid dimension
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

