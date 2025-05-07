package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.pooling.PoolingType;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class PoolingKernel extends PoolingBaseKernel {
    private PoolingType type;
    private int C;
    private int H;
    private int W;
    private int ph;
    private int pw;
    private int s;
    private int padding = 0;
    private int oHeight;
    private int oWidth;
    private int numKernels;
    private int max_f_n = 0;
    private int max_b_n = 0;
    private CUfunction forward_function;
    private CUfunction backward_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer dm;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public PoolingKernel(PoolingType type, int C, int H, int W, int ph, int pw, int s, CUDAManager cudaManager) {
        super(cudaManager);
        this.type = type;
        this.C = C;
        this.H = H;
        this.W = W;
        this.ph = ph;
        this.pw = pw;
        this.s = s;
        this.oHeight = (H - ph) / s + 1;
        this.oWidth = (W - pw) / s + 1;
        this.numKernels = 0;
        init();
    }

    public PoolingKernel(PoolingType type, int C, int H, int W, int ph, int pw, int s, int p, CUDAManager cudaManager) {
        super(cudaManager);
        this.type = type;
        this.C = C;
        this.H = H;
        this.W = W;
        this.ph = ph;
        this.pw = pw;
        this.s = s;
        this.padding = p;
        this.oHeight = (H + p - ph) / s + 1;
        this.oWidth = (W + p - pw) / s + 1;
        this.numKernels = 0;
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 3;
        int H = 4;
        int W = 4;
        int ph = 4;
        int pw = 4;
        int s = 4;
        int oHeight = (H - ph) / s + 1;
        int oWidth = (W - pw) / s + 1;
        float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        float[] d = RandomUtils.order(N * C * oHeight * oWidth, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor output = new Tensor(N, C, oHeight, oWidth, true);
        Tensor delta = new Tensor(N, C, oHeight, oWidth, d, true);
        Tensor diff = new Tensor(N, C, H, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        PoolingKernel pooling = new PoolingKernel(PoolingType.MEAN_POOLING, C, H, W, ph, pw, s, cudaManager);
        long start = System.nanoTime();
        for (int i = 0; i < 2; i++) {
            //    		output.clearGPU();
            pooling.forward(input, output);
        }
        System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
        output.showDM();
        input.showDM();
        pooling.backward(input, output, delta, diff);
        delta.showDM();
        diff.showDM();
        //    	System.out.println(JsonUtils.toJson(out));
        //    	System.out.println(JsonUtils.toJson(mask));
        //	    System.out.println(JsonUtils.toJson(out));
        //
        //	    System.out.println(JsonUtils.toJson(x));
        //
        //	    System.out.println(JsonUtils.toJson(xout));
        //
        //    	Tensor test = new Tensor(N, C, oHeight, oWidth, output.data, true);
        //
        //    	test.clearGPU();
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                switch (type) {
                    case MAX_POOLING:
                        forward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "maxpool_forward");
                        break;
                    case MEAN_POOLING:
                        forward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "meanpool_forward");
                        break;
                    case AVG_POOLING:
                        forward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "avgpool_forward");
                        break;
                }
            }
            if (backward_function == null) {
                switch (type) {
                    case MAX_POOLING:
                        backward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "maxpool_backward");
                        break;
                    case MEAN_POOLING:
                        backward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "meanpool_backward");
                        break;
                    case AVG_POOLING:
                        backward_function = getCudaManager().getLocalFunctionByModule("PoolingV2Kernel.cu", "avgpool_backward");
                        break;
                }
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

    public void forward(Tensor input, Tensor output) {
        output.clearGPU();
        switch (type) {
            case MAX_POOLING:
                maxpooling(input, output);
                break;
            case MEAN_POOLING:
                meanpooling(input, output);
                break;
        }
    }

    public void backward(Tensor input, Tensor output, Tensor delta, Tensor diff) {
        diff.clearGPU();
        switch (type) {
            case MAX_POOLING:
                maxpoolingDiff(delta, diff);
                break;
            case MEAN_POOLING:
                meanpoolingDiff(delta, diff);
                break;
        }
    }

    public void maxpooling(Tensor input, Tensor output) {
        try {
            //			long start1 = System.nanoTime();
            if (this.dm == null || input.number != this.N) {
                /**
                 * 申请显存

                 */
                this.dm = CUDAMemoryManager.getPointer(input.number * C * oHeight * oWidth, Sizeof.INT);
                this.N = input.number;
                this.max_f_n = oHeight * oWidth * C * N;
                /**
                 * 设置入参
                 * int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes

                 */
                forwardKernelParameters = Pointer.to(Pointer.to(new int[]{max_f_n}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{C}), Pointer.to(new int[]{s}), Pointer.to(new int[]{ph}), Pointer.to(new int[]{padding}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(dm));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(max_f_n), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //
            //			int[] tmp = new int[input.number * C * oHeight * oWidth];
            //
            //			this.showDM(dm, tmp);
            //
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void meanpooling(Tensor input, Tensor output) {
        try {
            //			long start1 = System.nanoTime();
            if (input.number != this.N) {
                this.N = input.number;
                this.max_f_n = oHeight * oWidth * C * N;
                /**
                 * 设置入参
                 * int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output

                 */
                forwardKernelParameters = Pointer.to(Pointer.to(new int[]{max_f_n}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{C}), Pointer.to(new int[]{s}), Pointer.to(new int[]{ph}), Pointer.to(new int[]{padding}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(max_f_n), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void avgpooling(Tensor input, Tensor output) {
        try {
            //			long start1 = System.nanoTime();
            if (input.number != this.N) {
                this.N = input.number;
                this.numKernels = this.N * input.channel;
                /**
                 * 设置入参
                 * int n, int w, int h, int c, float *input, float *output

                 */
                forwardKernelParameters = Pointer.to(Pointer.to(new int[]{numKernels}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H}), Pointer.to(new int[]{C}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()));
            }
            cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(numKernels), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void maxpoolingDiff(Tensor delta, Tensor diff) {
        try {
            if (backwardKernelParameters == null) {
                this.max_b_n = H * W * C * N;
                /**
                 * 设置入参
                 * int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes

                 */
                backwardKernelParameters = Pointer.to(Pointer.to(new int[]{max_b_n}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{C}), Pointer.to(new int[]{s}), Pointer.to(new int[]{ph}), Pointer.to(new int[]{padding}), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(dm));
            }
            cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(max_b_n), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void meanpoolingDiff(Tensor delta, Tensor diff) {
        try {
            if (backwardKernelParameters == null) {
                this.max_b_n = H * W * C * N;
                /**
                 * 设置入参
                 * int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta

                 */
                backwardKernelParameters = Pointer.to(Pointer.to(new int[]{max_b_n}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{C}), Pointer.to(new int[]{s}), Pointer.to(new int[]{ph}), Pointer.to(new int[]{padding}), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()));
            }
            cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(max_b_n), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void avgpoolingDiff(Tensor delta, Tensor diff) {
        try {
            //			long start1 = System.nanoTime();
            if (backwardKernelParameters == null) {
                /**
                 * 设置入参
                 * int n, int w, int h, int c, float *in_delta, float *out_delta

                 */
                backwardKernelParameters = Pointer.to(Pointer.to(new int[]{numKernels}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H}), Pointer.to(new int[]{C}), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()));
            }
            cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(numKernels), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void poolingDiff(Tensor delta, Tensor diff, int index) {
        try {
            //			long start1 = System.nanoTime();
            /**
             * 设置入参
             * float* x,float* mask,float* result,int n,int height,int width,int oHeight,int oWidth,int pWidth,int pHeight,int stride

             */
            backwardKernelParameters = Pointer.to(Pointer.to(delta.getGpuData().withByteOffset(index * C * oHeight * oWidth * Sizeof.FLOAT)), Pointer.to(dm.withByteOffset(index * C * oHeight * oWidth * ph * pw * Sizeof.FLOAT)), Pointer.to(diff.getGpuData().withByteOffset(index * C * H * W * Sizeof.FLOAT)), Pointer.to(new int[]{numKernels}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{ph}), Pointer.to(new int[]{pw}), Pointer.to(new int[]{s}));
            cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(numKernels), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
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

