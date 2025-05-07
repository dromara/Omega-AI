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

public class BaseKernel {
    public int N = 0;
    public int BN = 0;
    private CUDAManager cudaManager;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private CUfunction copy_gpu_function;
    private CUfunction axpy_gpu_function;
    private CUfunction fill_gpu_function;
    private CUfunction scal_add_function;
    private CUfunction constrain_function;
    private CUfunction concat_channel_function;
    private CUfunction concat_channel_backward_function;
    private CUfunction replace_channel_forward_function;
    private CUfunction replace_channel_backward_function;
    private CUfunction addMul_function;
    private CUfunction unMul_function;
    private CUfunction unMul_grad_function;

    public BaseKernel(CUDAManager cudaManager) {
        this.cudaManager = cudaManager;
        copy_gpu_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "copy_kernel");
        axpy_gpu_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "axpy_kernel");
        fill_gpu_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "fill_kernel");
        scal_add_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "scal_add_kernel");
        constrain_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "constrain_kernel");
        concat_channel_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "concat_channel_forward_kernel");
        concat_channel_backward_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "concat_channel_backward_kernel");
        replace_channel_forward_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "replace_channel_forward_kernel");
        replace_channel_backward_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "replace_channel_backward_kernel");
        addMul_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "add_mul_kernel");
        unMul_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "un_mul_kernel");
        unMul_grad_function = cudaManager.getLocalFunctionByModule("BaseKernel.cu", "un_mul_grad_kernel");
    }

    public void concat_channel_forward(Tensor x1, Tensor x2, Tensor output, int B, int C1, int C2, int H, int W) {
        try {
            if (concat_channel_function == null) {
                concat_channel_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "concat_channel_forward_kernel");
            }
            /**
             *  const float* x1, const float* x2,float* out,int B, int C1, int C2, int H, int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{C1}), Pointer.to(new int[]{C2}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
            int N = B * (int) Math.max(C1, C2) * H * W;
            checkCUDA(cuLaunchKernel(concat_channel_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void concat_channel_backward(Tensor diff, Tensor dx1, Tensor dx2, int B, int C1, int C2, int H, int W) {
        try {
            if (concat_channel_backward_function == null) {
                concat_channel_backward_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "concat_channel_backward_kernel");
            }
            /**
             *   const float* dout, float* dx1, float* dx2,int B, int C1, int C2, int H, int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dx1.getGpuData()), Pointer.to(dx2.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{C1}), Pointer.to(new int[]{C2}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}));
            int N = B * (int) Math.max(C1, C2) * H * W;
            checkCUDA(cuLaunchKernel(concat_channel_backward_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void replace_channel_forward(Tensor x1, Tensor x2, Tensor output, Tensor indices, int size) {
        try {
            if (replace_channel_forward_function == null) {
                replace_channel_forward_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "replace_channel_forward_kernel");
            }
            /**
             *   const float* out,float* x1, float* x2,int B, int C, int H, int W,int N, int* indices,int size

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(new int[]{x1.number}), Pointer.to(new int[]{x1.channel}), Pointer.to(new int[]{x1.height}), Pointer.to(new int[]{x1.width}), Pointer.to(new int[]{x1.getDataLength()}), Pointer.to(indices.getGpuData()), Pointer.to(new int[]{size}));
            checkCUDA(cuLaunchKernel(replace_channel_forward_function, CAFFE_GET_BLOCKS(x1.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void replace_channel_forward(Tensor x1, Tensor x2, Tensor output, Tensor indices, int size, int B, int C, int H, int W) {
        try {
            if (replace_channel_forward_function == null) {
                replace_channel_forward_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "replace_channel_forward_kernel");
            }
            /**
             *   const float* out,float* x1, float* x2,int B, int C, int H, int W,int N, int* indices,int size

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{x1.getDataLength()}), Pointer.to(indices.getGpuData()), Pointer.to(new int[]{size}));
            checkCUDA(cuLaunchKernel(replace_channel_forward_function, CAFFE_GET_BLOCKS(x1.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void replace_channel_backward(Tensor diff, Tensor dx, Tensor indices, int size) {
        try {
            if (replace_channel_backward_function == null) {
                replace_channel_backward_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "replace_channel_backward_kernel");
            }
            /**
             *   const float* diff,float* dx,int B, int C, int H, int W,int N, int* indices,int size

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{diff.number}), Pointer.to(new int[]{diff.channel}), Pointer.to(new int[]{diff.height}), Pointer.to(new int[]{diff.width}), Pointer.to(new int[]{diff.getDataLength()}), Pointer.to(indices.getGpuData()), Pointer.to(new int[]{size}));
            checkCUDA(cuLaunchKernel(replace_channel_backward_function, CAFFE_GET_BLOCKS(diff.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void replace_channel_backward(Tensor diff, Tensor dx, Tensor indices, int size, int B, int C, int H, int W) {
        try {
            if (replace_channel_backward_function == null) {
                replace_channel_backward_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "replace_channel_backward_kernel");
            }
            /**
             *   const float* diff,float* dx,int B, int C, int H, int W,int N, int* indices,int size

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{diff.getDataLength()}), Pointer.to(indices.getGpuData()), Pointer.to(new int[]{size}));
            checkCUDA(cuLaunchKernel(replace_channel_backward_function, CAFFE_GET_BLOCKS(diff.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void constrain_gpu(int N, float ALPHA, Tensor a, int INCX) {
        try {
            if (constrain_function == null) {
                constrain_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "constrain_kernel");
            }
            /**
             * int N, float ALPHA, float *X, int INCX

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(new float[]{ALPHA}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{INCX}));
            checkCUDA(cuLaunchKernel(constrain_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void constrain_gpu(int N, float ALPHA, Tensor a, int INCX, int offset) {
        try {
            if (constrain_function == null) {
                constrain_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "constrain_kernel");
            }
            /**
             * int N, float ALPHA, float *X, int INCX

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(new float[]{ALPHA}), Pointer.to(a.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(new int[]{INCX}));
            checkCUDA(cuLaunchKernel(constrain_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fill_gpu(Tensor a, float val) {
        try {
            if (fill_gpu_function == null) {
                fill_gpu_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "fill_kernel");
            }
            /**
             * int N, float ALPHA, float *X, int INCX

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(new float[]{val}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{1}));
            checkCUDA(cuLaunchKernel(fill_gpu_function, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scal_add_gpu(Tensor a, int N, float ALPHA, float BETA, int OFFX, int INCX) {
        try {
            /**
             * int N, float ALPHA, float BETA, float *X, int INCX

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(new float[]{ALPHA}), Pointer.to(new float[]{BETA}), Pointer.to(a.getGpuData().withByteOffset(OFFX * Sizeof.FLOAT)), Pointer.to(new int[]{INCX}));
            checkCUDA(cuLaunchKernel(scal_add_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void axpy_gpu(Tensor a, Tensor b, int N, float ALPHA, int INCX, int INCY) {
        axpy_gpu(a, b, N, ALPHA, 0, INCX, 0, INCY);
    }

    public void axpy_gpu(Tensor a, Tensor b, int N, float ALPHA, int OFFX, int INCX, int OFFY, int INCY) {
        try {
            if (axpy_gpu_function == null) {
                axpy_gpu_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "axpy_kernel");
            }
            /**
             * int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(new float[]{ALPHA}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{OFFX}), Pointer.to(new int[]{INCX}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{OFFY}), Pointer.to(new int[]{INCY}));
            checkCUDA(cuLaunchKernel(axpy_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Pointer a, Pointer b, int N, int incx, int incy) {
        try {
            if (copy_gpu_function == null) {
                copy_gpu_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "copy_kernel");
            }
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a), Pointer.to(new int[]{0}), Pointer.to(new int[]{incx}), Pointer.to(b), Pointer.to(new int[]{0}), Pointer.to(new int[]{incy}));
            checkCUDA(cuLaunchKernel(copy_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Tensor a, Tensor b, int N, int incx, int incy) {
        try {
            if (copy_gpu_function == null) {
                copy_gpu_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "copy_kernel");
            }
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{0}), Pointer.to(new int[]{incx}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{0}), Pointer.to(new int[]{incy}));
            cuLaunchKernel(copy_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Tensor a, Tensor b, int N, int offx, int incx, int offy, int incy) {
        try {
            if (copy_gpu_function == null) {
                copy_gpu_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "copy_kernel");
            }
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{offx}), Pointer.to(new int[]{incx}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{offy}), Pointer.to(new int[]{incy}));
            cuLaunchKernel(copy_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_mul(Tensor a, Tensor b, Tensor input, Tensor noise, Tensor output) {
        try {
            if (addMul_function == null) {
                addMul_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "add_mul_kernel");
            }
            /**
             * float* input,float* noise,float* a,float* b,int N, int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{input.dataLength}), Pointer.to(new int[]{input.getOnceSize()}));
            cuLaunchKernel(addMul_function, CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void un_mul(Tensor a, Tensor b, Tensor input, Tensor noise, Tensor output) {
        try {
            if (unMul_function == null) {
                unMul_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "un_mul_kernel");
            }
            /**
             * float* input,float* noise,float* a,float* b,int N, int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{input.dataLength}), Pointer.to(new int[]{input.getOnceSize()}));
            cuLaunchKernel(unMul_function, CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void un_mul_grad(Tensor a, Tensor b, Tensor delta, Tensor noise, Tensor diff) {
        try {
            if (unMul_grad_function == null) {
                unMul_grad_function = getCudaManager().getLocalFunctionByModule("BaseKernel.cu", "un_mul_grad_kernel");
            }
            /**
             * float* input,float* noise,float* a,float* b,int N, int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{delta.dataLength}), Pointer.to(new int[]{delta.getOnceSize()}));
            cuLaunchKernel(unMul_grad_function, CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            );
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

    public CUDAManager getCudaManager() {
        return cudaManager;
    }
}

