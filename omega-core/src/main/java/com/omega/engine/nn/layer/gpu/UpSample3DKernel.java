package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class UpSample3DKernel extends BaseKernel {

    private CUfunction forward_function;
    private CUfunction backward_function;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public UpSample3DKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        try {
            int N = 2;
            int C = 3;
            int D = 2;
            int H = 2;
            int W = 2;
            int scale = 2;
            int oDepth = D * scale;
            int oHeight = H * scale;
            int oWidth = W * scale;
            float[] x = MatrixUtils.order(N * C * D * H * W, 1, 1);
            float[] d = RandomUtils.order(N * C * oDepth * oHeight * oWidth, 0.1f, 0.1f);
            Tensor input = new Tensor(N, C * D, H, W, x, true);
            Tensor output = new Tensor(N, C * oDepth, oHeight, oWidth, true);
            Tensor delta = new Tensor(N, C * oDepth, oHeight, oWidth, d, true);
            delta.showShape();
            Tensor diff = new Tensor(N, C * D, H, W, true);
            CUDAManager cudaManager = new CUDAManager(0);
            UpSample3DKernel pooling = new UpSample3DKernel(cudaManager);
            long start = System.nanoTime();
            //        	for(int i = 0;i<2;i++) {
            pooling.forward(input, output, C, D, H, W, scale);
            //        	}
            System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
            input.showDM();
            output.showDM();
            pooling.backward(delta, diff, C, D, H, W, scale);
            delta.showDM();
            diff.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "upscale3d");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "downscale3d");
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

    public void forward(Tensor input, Tensor output,int channel,int depth,int height,int width, int scale) {
        upsample(input, output, channel, depth, height, width, scale);
    }

    public void backward(Tensor delta, Tensor diff,int channel,int depth,int height,int width, int scale) {
        upsampleDelta(delta, diff, channel, depth, height, width, scale);
    }

    public void upsample(Tensor input, Tensor output,int channel,int depth,int height,int width, int scale) {
        try {
        	int d1 = channel;
        	int d2 = depth * scale;
        	int d3 = height * scale;
        	int d4 = width * scale;
            this.N = input.getShape()[0];

            /**
             * 设置入参
             * const float *input, float *output, int no_elements,int scale_factor, int d1, int d2, int d3, int d4
             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.getDataLength()}), Pointer.to(new int[]{scale}), Pointer.to(new int[]{d1}), Pointer.to(new int[]{d2}), Pointer.to(new int[]{d3}), Pointer.to(new int[]{d4}));
            int nthreads = 256;
            int n_xblocks = Math.min(Math.max((int) Math.ceil((float) output.getDataLength() / nthreads), 1), 65535);
            int n_yblocks = (int) Math.ceil((float) output.getDataLength() / (float) (n_xblocks * nthreads));
            int[] blocks = new int[]{n_xblocks, n_yblocks, 1};
            int[] threads = new int[]{nthreads, 1, 1};
            checkCUDA(cuLaunchKernel(forward_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void upsampleDelta(Tensor delta, Tensor diff,int channel,int depth,int height,int width, int scale) {
        try {
            diff.clearGPU();
            int d1 = channel;
        	int d2 = depth;
        	int d3 = height;
        	int d4 = width;
            this.N = delta.getShape()[0];
            /**
             * 设置入参
             * float *gradInput_data, const float *gradOutput_data, int no_elements, int scale_factor, int d1, int d2, int d3, int d4
             */
            backwardKernelParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{diff.getDataLength()}), Pointer.to(new int[]{scale}), Pointer.to(new int[]{d1}), Pointer.to(new int[]{d2}), Pointer.to(new int[]{d3}), Pointer.to(new int[]{d4}));
            int nthreads = 256;
            int n_xblocks = Math.min(Math.max((int) Math.ceil((float) diff.getDataLength() / nthreads), 1), 65535);
            int n_yblocks = (int) Math.ceil((float) diff.getDataLength() / (float) (n_xblocks * nthreads));
            int[] blocks = new int[]{n_xblocks, n_yblocks, 1};
            int[] threads = new int[]{nthreads, 1, 1};
            checkCUDA(cuLaunchKernel(backward_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            ));
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

