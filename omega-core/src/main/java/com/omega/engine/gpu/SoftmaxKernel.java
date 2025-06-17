package com.omega.engine.gpu;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class SoftmaxKernel extends BaseKernel {
    private CUfunction softmax_function;
    private CUfunction softmax_mask_function;
    private CUfunction log_softmax_function;
    private CUfunction softmax_backward_function;
    private CUfunction softmax_mask_backward_function;
    private CUfunction log_softmax_backward_function;
    private CUfunction log_softmax_backward_function2;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private Pointer kernelMaskParameters;
    private Pointer backKernelParameters;
    private Pointer backKernelParameters2;

    public SoftmaxKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    /**
     * bottom_diff = top_diff * (top_data - top_data * top_data)
     *
     * @param output
     * @param delta
     * @param diff
     */
    public static void cpuBackwardNoLoss(Tensor output, Tensor delta, Tensor diff) {
        // TODO Auto-generated method stub
        //		GPUOP.getInstance().bmm(delta.getGpuData(), output.getGpuData(), diff.getGpuData(), delta.number * delta.channel, delta.height, output.height, delta.width, CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
        //		Tensor tmp = new Tensor(delta.number, 1, 1, diff.width, true);
        //
        //		GPUOP.getInstance().multiplyFloat(delta.number, output.width, delta.width, delta.getGpuData(), output.getGpuData(), tmp.getGpuData(), CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
        //
        //		tmp.showDM();
        //
        //		tmp.syncHost();
        //
        //		for(int i = 0;i<output.getDataLength();i++) {
        //			int n = i / output.width;
        //		    int s = i % output.width;
        //			diff.data[i] = (delta.data[i] - tmp.data[n * output.width + s]) * output.data[i];
        //		}
        //		diff.hostToDevice();
        //		System.out.println(JsonUtils.toJson(diff.data));
        for (int n = 0; n < output.getShape()[0]; n++) {
            float sum = 0.0f;
            for (int w = 0; w < output.getShape()[3]; w++) {
                sum += output.getData()[n * output.getShape()[3] + w] * delta.getData()[n * output.getShape()[3] + w];
            }
            for (int w = 0; w < output.getShape()[3]; w++) {
                diff.getData()[n * output.getShape()[3] + w] = (delta.getData()[n * output.getShape()[3] + w] - sum) * output.getData()[n * output.getShape()[3] + w];
            }
        }
        diff.hostToDevice();
    }

    public static void safeSoftmax_3pass(Tensor input, Tensor output) {
        for (int id = 0; id < input.getShape()[0]; id++) {
            float max = -3.402823466e+38F;
            float sum = 0;
            for (int i = 0; i < input.getShape()[3]; i++) {
                if (max <= input.getData()[id * input.getShape()[3] + i]) {
                    max = input.getData()[id * input.getShape()[3] + i];
                }
            }
            for (int i = 0; i < input.getShape()[3]; i++) {
                float e = (float) Math.exp(input.getData()[id * input.getShape()[3] + i] - max);
                sum += e;
                output.getData()[id * input.getShape()[3] + i] = e;
            }
            for (int i = 0; i < input.getShape()[3]; i++) {
                output.getData()[id * input.getShape()[3] + i] /= sum;
            }
        }
    }

    public static void safeSoftmax_2pass(Tensor input, Tensor output) {
        for (int id = 0; id < input.getShape()[0]; id++) {
            float max = -3.402823466e+38F;
            float max_p = -3.402823466e+38F;
            float sum = 0;
            for (int i = 0; i < input.getShape()[3]; i++) {
                if (max <= input.getData()[id * input.getShape()[3] + i]) {
                    max = input.getData()[id * input.getShape()[3] + i];
                }
                float e = (float) Math.exp(input.getData()[id * input.getShape()[3] + i] - max);
                float e_p = (float) Math.exp(max_p - max);
                sum = sum * e_p + e;
                max_p = max;
            }
            for (int i = 0; i < input.getShape()[3]; i++) {
                float e = (float) Math.exp(input.getData()[id * input.getShape()[3] + i] - max);
                output.getData()[id * input.getShape()[3] + i] = e / sum;
            }
        }
    }

    public static void main(String[] args) {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 20000;
        float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x, true);
        //		input.showDM();
        Tensor output = new Tensor(N, C, H, W, true);
        long start1 = System.nanoTime();
        safeSoftmax_3pass(input, output);
        System.out.println((System.nanoTime() - start1) / 1e6 + "ms");
        output.hostToDevice();
        output.showDM();
        long start2 = System.nanoTime();
        safeSoftmax_2pass(input, output);
        System.out.println((System.nanoTime() - start2) / 1e6 + "ms");
        output.hostToDevice();
        output.showDM();
    }

    public void initFunction() {
        try {
            if (softmax_function == null) {
                softmax_function = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "softmax");
            }
            if (softmax_mask_function == null) {
                softmax_mask_function = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "softmax_mask");
            }
            if (log_softmax_function == null) {
                log_softmax_function = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "log_softmax");
            }
            if (softmax_backward_function == null) {
                softmax_backward_function = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "softmax_back");
            }
            if (log_softmax_backward_function == null) {
                log_softmax_backward_function = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "log_softmax_back");
            }
            if (log_softmax_backward_function2 == null) {
                log_softmax_backward_function2 = getCudaManager().getLocalFunctionByModule("SoftmaxKernel.cu", "log_softmax_back2");
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

    public void softmax(Tensor input, Tensor output) {
        if (kernelParameters == null || this.N != output.getShape()[0]) {
            /**
             * float *input, float *output, int batch, int n, float temp

             */
            kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.getShape()[0] * input.getShape()[1] * input.getShape()[2]}), Pointer.to(new int[]{input.getShape()[3]}));
            this.N = output.getShape()[0];
        }
        cuLaunchKernel(softmax_function, this.CAFFE_GET_BLOCKS(input.getShape()[0] * input.getShape()[1] * input.getShape()[2]), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void softmax_out(Tensor input, Tensor output) {
        /**
         * float *input, float *output, int batch, int n, float temp

         */
        kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.getShape()[0] * input.getShape()[1] * input.getShape()[2]}), Pointer.to(new int[]{input.getShape()[3]}));
        cuLaunchKernel(softmax_function, this.CAFFE_GET_BLOCKS(input.getShape()[0] * input.getShape()[1] * input.getShape()[2]), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void softmaxMask(Tensor input, Tensor mask, Tensor output, float tmp) {
        //		if(kernelMaskParameters == null || this.N != output.number) {
        /**
         * float *input, float *output, float *mask, int batch, int n, float tmp

         */
        kernelMaskParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(new int[]{input.getShape()[0] * input.getShape()[1] * output.getShape()[2]}), Pointer.to(new int[]{input.getShape()[3]}), Pointer.to(new float[]{tmp}));
        this.N = output.getShape()[0];
        //		}
        cuLaunchKernel(softmax_mask_function, this.CAFFE_GET_BLOCKS(input.getShape()[0] * input.getShape()[1] * output.getShape()[2]), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelMaskParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void log_softmax(Tensor input, Tensor output) {
        if (kernelParameters == null || this.N != output.getShape()[0]) {
            /**
             * float *input, float *output, int batch, int n, float temp

             */
            kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{input.getShape()[0]}), Pointer.to(new int[]{input.getShape()[3]}));
            this.N = output.getShape()[0];
        }
        cuLaunchKernel(log_softmax_function, input.getShape()[0], 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward_noloss(Tensor output, Tensor delta, Tensor diff) {
        if (backKernelParameters == null) {
            /**
             * float *output, float *delta, float *diff, int batch, int n

             */
            backKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{output.getShape()[0] * output.getShape()[1] * output.getShape()[2]}), Pointer.to(new int[]{output.getShape()[3]}));
        }
        cuLaunchKernel(softmax_backward_function, this.CAFFE_GET_BLOCKS(output.getShape()[0] * output.getShape()[1] * output.getShape()[2]), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                backKernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward(Tensor output, Tensor currentLabel, Tensor diff) {
        if (backKernelParameters == null) {
            /**
             * float* x,float* mean,float* var,int number,int channel,int height,int width

             */
            backKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{diff.getDataLength()}));
        }
        cuLaunchKernel(log_softmax_backward_function, this.CAFFE_GET_BLOCKS(diff.getDataLength()), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                backKernelParameters, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void backward2(Tensor output, Tensor currentLabel, Tensor diff) {
        if (backKernelParameters2 == null) {
            /**
             * float* x,float* mean,float* var,int number,int channel,int height,int width

             */
            backKernelParameters2 = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(currentLabel.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{diff.getDataLength()}), Pointer.to(new int[]{N}));
        }
        cuLaunchKernel(log_softmax_backward_function2, this.CAFFE_GET_BLOCKS(diff.getDataLength()), 1, 1,      // Grid dimension
                CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                backKernelParameters2, null // Kernel- and extra parameters
        );
        //		JCudaDriver.cuCtxSynchronize();
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    public void cpuForward(Tensor input, Tensor output) {
        float[] dest = new float[input.getShape()[1] * input.getShape()[2] * input.getShape()[3]];
        for (int n = 0; n < input.getShape()[0]; n++) {
            input.copy(n, dest);
            float max = MatrixOperation.max(dest);
            float[] temp = MatrixOperation.subtraction(dest, max);
            temp = MatrixOperation.exp(temp);
            float sum = MatrixOperation.sum(temp);
            for (int i = 0; i < temp.length; i++) {
                output.getData()[n * output.getShape()[1] * output.getShape()[2] * output.getShape()[3] + i] = temp[i] / sum;
            }
        }
        //		System.out.println(JsonUtils.toJson(output.data));
    }

    //	public static void main(String[] args) {
    //
    //		int N = 2;
    //		int C = 5;
    //		int H = 4;
    //		int W = 4;
    //
    //		float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
    //
    //		Tensor mask = ENTokenizer.triu(N, C, H, W, 1);
    //		mask.showDM();
    //
    //		Tensor input = new Tensor(N, C, H, W, x, true);
    //
    //		Tensor output = new Tensor(N, C, H, W, true);
    //
    //		Tensor output2 = new Tensor(N, C, H, W);
    //
    //		SoftmaxKernel k = new SoftmaxKernel();
    ////		k.softmax(input, output);
    //
    //		k.softmaxMask(input, mask, output, -1e9f);
    //
    //		output.showDM();
    //
    ////		k.cpuForward2(input, output2);
    //
    //		k.cpuForwardMask(input, output2, mask, -1e9f);
    //
    //		System.out.println("output2:"+JsonUtils.toJson(output2.data));
    //
    //		Tensor delta = new Tensor(N, C, H, W, RandomUtils.order(N * C * H * W, 0.1f, 0), true);
    //
    //		Tensor diff = new Tensor(N, C, H, W, true);
    //
    //		k.backward_noloss(output, delta, diff);
    //
    ////		cpuBackwardNoLoss(output, delta, diff);
    //
    //		diff.showDM();
    //
    //	}
    public void cpuForward2(Tensor input, Tensor output) {
        for (int id = 0; id < input.getShape()[0]; id++) {
            float max = -3.402823466e+38F;
            float sum = 0;
            for (int i = 0; i < input.getShape()[3]; i++) {
                if (max <= input.getData()[id * input.getShape()[3] + i]) {
                    max = input.getData()[id * input.getShape()[3] + i];
                }
            }
            for (int i = 0; i < input.getShape()[3]; i++) {
                float e = (float) Math.exp(input.getData()[id * input.getShape()[3] + i] - max);
                sum += e;
                output.getData()[id * input.getShape()[3] + i] = e;
            }
            for (int i = 0; i < input.getShape()[3]; i++) {
                output.getData()[id * input.getShape()[3] + i] /= sum;
            }
        }
    }

    public void cpuForwardMask(Tensor input, Tensor output, Tensor mask, float tmp) {
        int n = input.getShape()[2] * input.getShape()[3];
        for (int id = 0; id < input.getShape()[0] * input.getShape()[1]; id++) {
            float max = -3.402823466e+38F;
            float sum = 0;
            int b = id / input.getShape()[1];
            //			System.out.println(id+":"+b);
            for (int i = 0; i < n; i++) {
                float val = input.getData()[id * n + i];
                //				System.out.println(b * n + i+":"+mask.data[b * n + i]);
                if (mask.getData()[b * n + i] == 1) {
                    val = tmp;
                }
                if (max <= val) {
                    max = val;
                }
            }
            for (int i = 0; i < n; i++) {
                float val = input.getData()[id * n + i];
                if (mask.getData()[b * n + i] == 1) {
                    val = tmp;
                }
                float e = (float) Math.exp(val - max);
                sum += e;
                output.getData()[id * n + i] = e;
            }
            for (int i = 0; i < n; i++) {
                output.getData()[id * n + i] /= sum;
            }
        }
    }

    public void cpuBackward(Tensor output, Tensor currentLabel, Tensor diff) {
        // TODO Auto-generated method stub
        for (int i = 0; i < output.getDataLength(); i++) {
            diff.getData()[i] = output.getData()[i] - currentLabel.getData()[i];
        }
        //		System.out.println(JsonUtils.toJson(diff.data));
    }
}

