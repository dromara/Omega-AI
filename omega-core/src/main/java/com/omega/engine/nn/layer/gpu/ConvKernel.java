package com.omega.engine.nn.layer.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ConvKernel extends ConvBaseKernel {
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
    private boolean is_1x1 = false;
    private int numKernels;
    private CUfunction im2col_function;
    private CUfunction bias_function;
    private CUfunction back_back_function;
    private CUfunction col2im_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private Pointer biasKernelParameters;
    private Pointer biasBackKernelParameters;
    private Pointer col2imKernelParameters;
    private Pointer dy;
    private Pointer dx_t;

    public ConvKernel(int C, int H, int W, int ko, int kh, int kw, int s, int p, CUDAManager cudaManager) {
        super(cudaManager);
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
        this.ih = C * kh * kw;
        this.iw = oHeight * oWidth;
        this.numKernels = C * oHeight * oWidth;
        if (kh == 1 && kw == 1 && s == 1 && p == 0) {
            is_1x1 = true;
        }
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 64;
        int H = 8;
        int W = 8;
        int ko = 128;
        int kh = 1;
        int kw = 1;
        int s = 2;
        int p = 0;
        int oHeight = ((H + 2 * p - kh) / s) + 1;
        int oWidth = ((W + 2 * p - kw) / s) + 1;
        float[] x1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        float[] k1 = RandomUtils.order(ko * C * kh * kw, 0.1f, 0.1f);
        float[] d1 = RandomUtils.order(N * ko * oHeight * oWidth, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x1, true);
        Tensor delta = new Tensor(N, ko, oHeight, oWidth, d1, true);
        Tensor kernel = new Tensor(ko, C, kh, kw, k1, true);
        Tensor output = new Tensor(N, ko, oHeight, oWidth, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        Tensor diffW = new Tensor(ko, C, kh, kw, true);
        CUDAManager cudaManager = new CUDAManager(0);
        ConvKernel k = new ConvKernel(C, H, W, ko, kh, kw, s, p, cudaManager);
        k.conv(input, kernel, output);
        output.syncHost();
        System.out.println("output:" + JsonUtils.toJson(output.getData()));
        k.dw(input, delta, diffW);
        diffW.syncHost();
        System.out.println("diffW:" + JsonUtils.toJson(diffW.getData()));
        k.dx(delta, kernel, diff);
        diff.syncHost();
        System.out.println("diff:" + JsonUtils.toJson(diff.getData()));
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
        if (!is_1x1) {
            this.dy = CUDAMemoryManager.getPointer(ih * iw);
            this.dx_t = CUDAMemoryManager.getPointer(ih * iw);
        }
    }

    public void initFunction() {
        try {
            if (im2col_function == null) {
                im2col_function = getCudaManager().getLocalFunctionByModule("Im2colKernel.cu", "im2col_gpu_kernelV2");
            }
            if (bias_function == null) {
                bias_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "add_bias");
            }
            if (back_back_function == null) {
                back_back_function = getCudaManager().getLocalFunctionByModule("BiasKernel.cu", "backward_bias_kernel");
            }
            if (col2im_function == null) {
                col2im_function = getCudaManager().getLocalFunctionByModule("Col2imKernel.cu", "col2im_gpu_kernelV2");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void conv(Tensor input, Tensor kernel, Tensor output) {
        for (int i = 0; i < input.getShape()[0]; i++) {
            if (!is_1x1) {
                im2col(input, i);
            } else {
                dy = input.getGpuData().withByteOffset(i * C * H * W * Sizeof.FLOAT);
            }
            sgemm(kernel, output, i);
        }
    }

    public void dw(Tensor input, Tensor delta, Tensor diffW) {
        diffW.clearGPU();
        for (int i = 0; i < input.getShape()[0]; i++) {
            if (!is_1x1) {
                im2col(input, i);
            } else {
                dy = input.getGpuData().withByteOffset(i * C * H * W * Sizeof.FLOAT);
            }
            sgemmDW(delta, diffW, i);
        }
    }

    public void dx(Tensor delta, Tensor kernel, Tensor diff) {
        for (int i = 0; i < delta.getShape()[0]; i++) {
            if (!is_1x1) {
                sgemmDX(delta, kernel, dx_t, i);
                col2im(diff, i);
            } else {
                sgemmDX(delta, kernel, diff.getGpuData().withByteOffset(i * ih * iw * Sizeof.FLOAT), i);
            }
        }
    }

    public void im2col(Tensor input, int index) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow

             */
            kernelParameters = Pointer.to(Pointer.to(input.getGpuData().withByteOffset(index * C * H * W * Sizeof.FLOAT)), Pointer.to(dy), Pointer.to(new int[]{numKernels}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{kh}), Pointer.to(new int[]{kw}), Pointer.to(new int[]{s}), Pointer.to(new int[]{p}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}));
            cuLaunchKernel(im2col_function, this.CAFFE_GET_BLOCKS(numKernels), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void col2im(Tensor diff, int index) {
        try {
            /**
             * 设置入参
             * float* data_col,float* data_im,int n,int height,int width,int channels,int ksize,int pad,int stride,int height_col,int width_col

             */
            col2imKernelParameters = Pointer.to(Pointer.to(dx_t), Pointer.to(diff.getGpuData().withByteOffset(index * C * H * W * Sizeof.FLOAT)), Pointer.to(new int[]{C * H * W}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{C}), Pointer.to(new int[]{kh}), Pointer.to(new int[]{kw}), Pointer.to(new int[]{p}), Pointer.to(new int[]{s}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}));
            checkCUDA(cuLaunchKernel(col2im_function, this.CAFFE_GET_BLOCKS(C * H * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    col2imKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sgemm(Tensor kernel, Tensor output, int index) {
        /**
         * m k n

         */
        getCudaManager().getOp().multiplyFloat(ko, iw, ih, kernel.getGpuData(), dy, output.getGpuData().withByteOffset(index * ko * oHeight * oWidth * Sizeof.FLOAT), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
    }

    public void sgemmDW(Tensor delta, Tensor diffW, int index) {
        /**
         * m k n

         */
        getCudaManager().getOp().multiplyFloat(ko, ih, iw, delta.getGpuData().withByteOffset(index * ko * iw * Sizeof.FLOAT), dy, diffW.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 1.0f);
    }

    public void sgemmDX(Tensor delta, Tensor kernel, Pointer p, int index) {
        /**
         * k n m

         */
        getCudaManager().getOp().multiplyFloat(ih, iw, ko, kernel.getGpuData(), delta.getGpuData().withByteOffset(index * ko * iw * Sizeof.FLOAT), p, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
    }

    public void addBias(Tensor output, Tensor bias) {
        try {
            if (biasKernelParameters == null || output.getShape()[0] != this.N) {
                /**
                 * 设置入参
                 * float* output, float* biases, int batch, int n, int size

                 */
                biasKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(new int[]{output.getNumber()}), Pointer.to(new int[]{output.getShape()[1]}), Pointer.to(new int[]{output.getShape()[2] * output.getShape()[3]}));
                this.N = output.getShape()[0];
            }
            cuLaunchKernel(bias_function, this.CAFFE_GET_BLOCKS(output.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    biasKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardBias(Tensor diffB, Tensor delta) {
        try {
            diffB.clearGPU();
            if (biasBackKernelParameters == null) {
                /**
                 * 设置入参
                 * float *bias_updates, float *delta, int batch, int n, int size

                 */
                biasBackKernelParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getNumber()}), Pointer.to(new int[]{delta.getChannel()}), Pointer.to(new int[]{delta.getShape()[2] * delta.getShape()[3]}));
            }
            cuLaunchKernel(back_back_function, delta.getChannel(), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    biasBackKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
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

    public void showDM(Pointer d, float[] data) {
        JCuda.cudaMemcpy(Pointer.to(data), d, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        System.out.println(JsonUtils.toJson(data));
    }

    @Override
    public void convTranspose(Tensor input, Tensor kernel, Tensor output) {
        // TODO Auto-generated method stub
    }
}

