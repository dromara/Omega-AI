package com.omega.engine.nn.layer.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.network.Transformer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AttentionKernel extends BaseKernel {
    private CUfunction permute_function;
    private CUfunction unpermute_function;
    private CUfunction permute_backward_function;
    private CUfunction unpermute_backward_function;
    private CUfunction softmax_forward_function;
    private CUfunction softmax_unmask_forward_function;
    private CUfunction softmax_test_forward_function;
    private CUfunction softmax_scale_test_forward_function;
    private CUfunction softmax_unmask_test_forward_function;
    private CUfunction softmax_backward_function;
    private CUfunction softmax_unmask_backward_function;
    private CUfunction softmax_test_backward_function;
    private CUfunction scale_function;
    private CUfunction add_mask_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private int BLOCK = 512;
    private Pointer permuteKernelParameters;
    private Pointer unpermuteKernelBackParameters;
    private Pointer permute_backwardKernelParameters;
    private Pointer unpermute_backwardKernelBackParameters;
    private Pointer softmaxForwardParameters;
    private Pointer softmaxBackwardParameters;
    private Pointer scaleParameters;
    private Pointer addMaskParameters;
    private CUfunction softmax_forward_2_function;
    private CUfunction softmax_backward_2_function;
    private CUfunction softmax_backward_unmask_2_function;
    private CUfunction softmax_backward_unmask_kv_function;

    public AttentionKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 4;
        int NH = 8;
        int T = 64;
        int T2 = 77;
        Transformer tf = new Transformer();
        tf.CUDNN = true;
        //		float[] x_data = RandomUtils.gaussianRandom(N * NH * T * T2, 0.1f);
        float[] x_data = MatrixUtils.order(N * NH * T * T2, 0.1f, 0.1f);
        Tensor x = new Tensor(N, NH, T, T2, x_data, true);
        Tensor delta = new Tensor(N, NH, T, T2, x_data, true);
        Tensor x2 = new Tensor(N, NH, T, T2, x_data, true);
        Tensor output = new Tensor(N, NH, T, T2, true);
        Tensor output2 = new Tensor(N, NH, T, T2, true);
        Tensor diff = new Tensor(N, NH, T, T2, true);
        //		Tensor output3 = new Tensor(N, NH, T, T, true);
        //
        //		Tensor datt = new Tensor(N, NH, T, T, x_data, true);
        //
        //		Tensor dpreatt = new Tensor(N, NH, T, T, true);
        AttentionKernel kernel = new AttentionKernel(tf.cudaManager);
        SoftmaxCudnnKernel cudnnKernel = new SoftmaxCudnnKernel(T2, 1, 1, tf.cudaManager);
        //		kernel.softmax_forward(x, output2, N, NH, T, 1);
        long start1 = System.nanoTime();
        kernel.softmax_test_forward(x, output, N, NH, T, T2);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start1) / 1e6 + "ms.");
        x2.view(N * NH * T, 1, 1, T2);
        long start2 = System.nanoTime();
        cudnnKernel.softmax(x2, output2);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start2) / 1e6 + "ms.");
        //		kernel.softmax_unmask_forward(x, output2, N, NH, T, 1f);
        //
        //		kernel.softmax_unmask_backward(dpreatt, datt, output, N, T, NH, 1);
        long start3 = System.nanoTime();
        kernel.softmax_test_forward(x, output, N, NH, T, T2);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start3) / 1e6 + "ms.");
        long start4 = System.nanoTime();
        cudnnKernel.softmax(x2, output2);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start4) / 1e6 + "ms.");
        output.showDMByOffset(0, 100);
        output.showShape();
        output2.showDMByOffset(0, 100);
        output2.showShape();
        long start5 = System.nanoTime();
        kernel.softmax_kv_unmask_backward(delta, output2, N, T, T2, NH, 1);
        //		cudnnKernel.softmax_backward(output2, x, diff);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start5) / 1e6 + "ms.");
        long start6 = System.nanoTime();
        cudnnKernel.softmax_backward(output2, x, x);
        JCuda.cudaDeviceSynchronize();
        System.out.println((System.nanoTime() - start6) / 1e6 + "ms.");
        x.showDMByOffset(0, 100);
        delta.showDMByOffset(0, 100);
        //		int batchSize = 2;
        //		int headNum = 3;
        //		int dk = 4;
        //		int time = 6;
        //		int kvTime = 7;
        //
        //		float[] qd = MatrixUtils.order(batchSize * headNum * time * dk, 0.1f, 0.1f);
        //
        //		float[] kd = MatrixUtils.order(batchSize * headNum * kvTime * dk, 0.1f, 0.1f);
        //
        //		Tensor query =  new Tensor(batchSize, headNum, time, dk, qd, true);
        //
        //		Tensor key =  new Tensor(batchSize, headNum, kvTime, dk, kd, true);
        //
        //		Tensor value =  new Tensor(batchSize, headNum, kvTime, dk, kd, true);
        //
        //		Tensor preatt = new Tensor(batchSize, headNum, time, kvTime, true);
        //
        //		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, key.getGpuData(), dk, kvTime * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        //		preatt.showShape();
        //		preatt.showDM();
        //
        //		Tensor vaccum = new Tensor(batchSize, headNum, time, dk, true);
        //
        //		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, kvTime, 1.0f, value.getGpuData(), dk, kvTime * dk, preatt.getGpuData(), kvTime, time * kvTime, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
        //
        //		vaccum.showDM();
        //
        //		Tensor vt =  new Tensor(batchSize, headNum, kvTime, dk, kd, true);
        //
        //		Tensor dvaccum =  new Tensor(batchSize, headNum, time, dk, qd, true);
        //
        //		Tensor dattn =  new Tensor(batchSize, headNum, time, kvTime, true);
        //
        //		/**
        //		 * backward into dattn[b, nh, t, t2]
        //		 * vt[b, nh, t2, dk] -> [b, nh, dk, t2]
        //		 * dvaccum[b, nh, t, dk]
        //		 */
        //		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, vt.getGpuData(), dk, kvTime * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        //
        //		dattn.showDM();
        //
        //		float[] attnd = MatrixUtils.order(batchSize * headNum * kvTime * time, 0.1f, 0.1f);
        //
        //		Tensor attn = new Tensor(batchSize, headNum, time, kvTime, attnd, true);
        //
        //		Tensor dvt =  new Tensor(batchSize, headNum, kvTime, dk, true);
        //
        //		/**
        //		 * backward into dvt[b, nh, t2, dk]
        //		 * dvaccum[b, nh, t, dk]
        //		 * attn[b, nh, t, t2] -> [b, nh, t2, t]
        //		 */
        //		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, kvTime, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), kvTime, time * kvTime, 0.0f, dvt.getGpuData(), dk, kvTime * dk, batchSize * headNum);
        //
        //		dvt.showDM();
        //		output2.showDM();
        //		PrintUtils.printImage(output2);
        //		System.err.println("======================================");
        //		PrintUtils.printImage(output);
        //		System.err.println("======================================");
        //		PrintUtils.printImage(dpreatt);
        //
        //		kernel.softmax_unmask_test_forward(x, output3, N, NH, T, 0.1f);
        //		output.showDM();
        //		output3.showDM();
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public void initFunction() {
        try {
            if (permute_function == null) {
                permute_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "permute_kernel");
            }
            if (unpermute_function == null) {
                unpermute_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "unpermute_kernel");
            }
            if (permute_backward_function == null) {
                permute_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "permute_kernel_backward");
            }
            if (unpermute_backward_function == null) {
                unpermute_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "unpermute_kernel_backward");
            }
            if (softmax_forward_function == null) {
                //				softmax_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel");
                softmax_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel5");
            }
            if (softmax_unmask_forward_function == null) {
                softmax_unmask_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel5_no_mask");
            }
            if (softmax_test_forward_function == null) {
                //				softmax_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel");
                softmax_test_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel4");
            }
            if (softmax_scale_test_forward_function == null) {
                softmax_scale_test_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_scale_forward_kernel4");
            }
            if (softmax_unmask_test_forward_function == null) {
                softmax_unmask_test_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_unmask_forward_kernel4");
            }
            if (softmax_backward_function == null) {
                //				softmax_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_kernel");
                softmax_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_kernel8");
            }
            if (softmax_unmask_backward_function == null) {
                softmax_unmask_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_nomask_backward_kernel8");
            }
            if (softmax_test_backward_function == null) {
                //				softmax_forward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel");
                softmax_test_backward_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_kernel4");
            }
            if (softmax_forward_2_function == null) {
                softmax_forward_2_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel52");
            }
            if (softmax_backward_2_function == null) {
                softmax_backward_2_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_inplace_kernel");
            }
            if (softmax_backward_unmask_2_function == null) {
                softmax_backward_unmask_2_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_unmask_backward_inplace_kernel");
            }
            if (softmax_backward_unmask_kv_function == null) {
                softmax_backward_unmask_kv_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "softmax_kv_unmask_backward_inplace_kernel");
            }
            if (scale_function == null) {
                scale_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "scale_kernel");
            }
            if (add_mask_function == null) {
                add_mask_function = getCudaManager().getLocalFunctionByModule("AttentionKernel.cu", "add_mask");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scale(Tensor input, float scale, int B, int NH, int T) {
        try {
            /**
             * 设置入参
             * float* inp, float scale, int B, int NH, int T

             */
            scaleParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(new float[]{scale}), Pointer.to(new int[]{B}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{T}));
            int total_threads = B * NH * T * T;
            int num_blocks = get_number_of_blocks(total_threads, BLOCK);
            checkCUDA(cuLaunchKernel(scale_function, num_blocks, 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    scaleParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void addMask(Tensor input, Tensor mask, Tensor output) {
        try {
            /**
             * 设置入参
             * int N, int C,int H,int W, float *input, float *mask,float *output

             */
            addMaskParameters = Pointer.to(Pointer.to(new int[]{input.dataLength}), Pointer.to(new int[]{input.channel}), Pointer.to(new int[]{input.height}), Pointer.to(new int[]{input.width}), Pointer.to(input.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(output.getGpuData()));
            checkCUDA(cuLaunchKernel(add_mask_function, this.CAFFE_GET_BLOCKS(input.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null,               // Shared memory size and stream
                    addMaskParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void permute(Tensor input, Tensor query, Tensor key, Tensor value, int B, int N, int NH, int d) {
        try {
            /**
             * 设置入参
             * float* q, float* k, float* v, const float* inp,int B, int N, int NH, int d

             */
            permuteKernelParameters = Pointer.to(Pointer.to(query.getGpuData()), Pointer.to(key.getGpuData()), Pointer.to(value.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{d}));
            int total_threads = B * NH * N * d;
            int num_blocks = get_number_of_blocks(total_threads, BLOCK);
            checkCUDA(cuLaunchKernel(permute_function, num_blocks, 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    permuteKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void permute_backward(Tensor diff, Tensor dQuery, Tensor dKey, Tensor dValue, int B, int N, int NH, int d) {
        try {
            /**
             * 设置入参
             * float* dinp, const float* dq, const float* dk, const float* dv,int B, int N, int NH, int d

             */
            permute_backwardKernelParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dQuery.getGpuData()), Pointer.to(dKey.getGpuData()), Pointer.to(dValue.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{d}));
            int total_threads = B * NH * N * d;
            int num_blocks = get_number_of_blocks(total_threads, BLOCK);
            checkCUDA(cuLaunchKernel(permute_backward_function, num_blocks, 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    permute_backwardKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void unpermute(Tensor input, Tensor output, int B, int N, int NH, int d) {
        try {
            /**
             * 设置入参
             * const float* inp, float *out, int B, int N, int NH, int d

             */
            unpermuteKernelBackParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{d}));
            int total_threads = output.getDataLength();
            int num_blocks = get_number_of_blocks(total_threads, BLOCK);
            checkCUDA(cuLaunchKernel(unpermute_function, num_blocks, 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    unpermuteKernelBackParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void unpermute_backward(Tensor din, Tensor dout, int B, int N, int NH, int d) {
        try {
            /**
             * 设置入参
             * float* dinp, const float *dout, int B, int N, int NH, int d

             */
            unpermute_backwardKernelBackParameters = Pointer.to(Pointer.to(din.getGpuData()), Pointer.to(dout.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{N}), Pointer.to(new int[]{NH}), Pointer.to(new int[]{d}));
            int total_threads = B * N * NH * d;
            int num_blocks = get_number_of_blocks(total_threads, BLOCK);
            checkCUDA(cuLaunchKernel(unpermute_backward_function, num_blocks, 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    unpermute_backwardKernelBackParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_forward(Tensor input, Tensor output, int B, int NH, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* out, float inv_temperature, const float* inp, int N, int T
             * float* out, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(new float[]{scale}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH}), Pointer.to(new int[]{T}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            //		    int grid_size = B * NH * T;
            int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            //		    int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax_unmask_forward(Tensor input, Tensor output, int B, int NH, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* out, float inv_temperature, const float* inp, int N, int T
             * float* out, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(new float[]{scale}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH}), Pointer.to(new int[]{T}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            //		    int grid_size = B * NH * T;
            int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            //		    int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_unmask_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax_unmask_forward(Tensor input, Tensor output, int B, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* out, float inv_temperature, const float* inp, int N, int T
             * float* out, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(new float[]{scale}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            //		    int grid_size = B * NH * T;
            int grid_size = (int) Math.ceil(B * T * 32 / softmax_block_size);
            //		    int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_unmask_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_test_forward(Tensor input, Tensor output, int B, int NH, int T) {
        try {
            /**
             * 设置入参
             * float* out,float scale, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()),
                    //	        	    Pointer.to(new float[]{scale}),
                    Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH * T}), Pointer.to(new int[]{T}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            int grid_size = B * NH * T;
            //		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_test_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_test_forward(Tensor input, Tensor output, int B, int NH, int T, int T2) {
        try {
            /**
             * 设置入参
             * float* out,float scale, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()),
                    //	        	    Pointer.to(new float[]{scale}),
                    Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH * T}), Pointer.to(new int[]{T2}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            int grid_size = B * NH * T;
            //		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_test_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_test_forward(Tensor input, Tensor output, int B, int NH, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* out,float scale, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()),
                    //	        	    Pointer.to(new float[]{scale}),
                    Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH * T}), Pointer.to(new int[]{T}), Pointer.to(new float[]{scale}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            int grid_size = B * NH * T;
            //		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_scale_test_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_unmask_test_forward(Tensor input, Tensor output, int B, int NH, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* out,float scale, const float* inp, int N, int C

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()),
                    //	        	    Pointer.to(new float[]{scale}),
                    Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH * T}), Pointer.to(new int[]{T}), Pointer.to(new float[]{scale}));
            //			int softmax_block_size = 256;
            //		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
            int softmax_block_size = 256;
            int grid_size = B * NH * T;
            //		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
            checkCUDA(cuLaunchKernel(softmax_unmask_test_forward_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_backward(Tensor dpreatt, Tensor datt, Tensor att, int B, int T, int C, int NH, float scale) {
        try {
            /**
             * 设置入参
             * float* dpreatt, const float* datt, const float* att,

             int B, int T, int C, float scale, int BlockSize

             */
            int block_size = 32;
            softmaxBackwardParameters = Pointer.to(Pointer.to(dpreatt.getGpuData()), Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new int[]{C}), Pointer.to(new float[]{scale}));
            //			int num_blocks = get_number_of_blocks(32/8*T, block_size);
            //	        int num_blocks = get_number_of_blocks(T, block_size);
            checkCUDA(cuLaunchKernel(softmax_backward_function, T / 4, B * NH, 1,      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax_unmask_backward(Tensor dpreatt, Tensor datt, Tensor att, int B, int T, int NH, float scale) {
        try {
            /**
             * 设置入参
             * float* dpreatt, const float* datt, const float* att,

             int B, int T, int C, float scale, int BlockSize

             */
            int block_size = 32;
            softmaxBackwardParameters = Pointer.to(Pointer.to(dpreatt.getGpuData()), Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new float[]{scale}));
            //			int num_blocks = get_number_of_blocks(32/8*T, block_size);
            //	        int num_blocks = get_number_of_blocks(T, block_size);
            checkCUDA(cuLaunchKernel(softmax_unmask_backward_function, T / 4, B * NH, 1,      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax_unmask_backward(Tensor dpreatt, Tensor datt, Tensor att, int B, int T, float scale) {
        try {
            /**
             * 设置入参
             * float* dpreatt, const float* datt, const float* att,int B, int T, float scale

             */
            int block_size = 32;
            softmaxBackwardParameters = Pointer.to(Pointer.to(dpreatt.getGpuData()), Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new float[]{scale}));
            //			int num_blocks = get_number_of_blocks(32/8*T, block_size);
            //	        int num_blocks = get_number_of_blocks(T, block_size);
            checkCUDA(cuLaunchKernel(softmax_unmask_backward_function, T / 4, B, 1,      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * N = B * NH
     *
     * @param input
     * @param output
     * @param scale
     * @param B
     * @param T
     */
    public void softmax_test_backward(Tensor dpreatt, Tensor datt, Tensor att, int B, int T, int C, int NH) {
        try {
            /**
             * 设置入参
             * float* __restrict__ dpreatt, const float* __restrict__ datt, const float* __restrict__ att, int B, int T, int C, int NH

             */
            int block_size = 256;
            softmaxBackwardParameters = Pointer.to(Pointer.to(dpreatt.getGpuData()), Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new int[]{C}), Pointer.to(new int[]{NH}));
            int num_blocks = get_number_of_blocks(32 / 8 * T, block_size);
            //	        int num_blocks = get_number_of_blocks(T, block_size);
            checkCUDA(cuLaunchKernel(softmax_test_backward_function, num_blocks, B * NH, 1,      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax2_forward(Tensor input, Tensor output, float scale, int B, int NH, int T) {
        try {
            /**
             * 设置入参
             * float* out, float inv_temperature, const float* inp, int N, int T

             */
            softmaxForwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(new float[]{scale}), Pointer.to(input.getGpuData()), Pointer.to(new int[]{B * NH}), Pointer.to(new int[]{T}));
            int softmax_block_size = 256;
            int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
            checkCUDA(cuLaunchKernel(softmax_forward_2_function, grid_size, 1, 1,      // Grid dimension
                    softmax_block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxForwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax2_backward(Tensor datt, Tensor att, int B, int T, int C, int NH, float scale) {
        try {
            /**
             * 设置入参
             * float* datt, const float* att, int B, int T, int C, float scale

             */
            int block_size = 256;
            softmaxBackwardParameters = Pointer.to(Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new int[]{C}), Pointer.to(new float[]{scale}));
            int grid_size = get_number_of_blocks(T, 4);
            int[] grids = new int[]{grid_size, B * NH, 1};
            checkCUDA(cuLaunchKernel(softmax_backward_2_function, grids[0], grids[1], grids[2],      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax2_unmask_backward(Tensor datt, Tensor att, int B, int T, int C, int NH, float scale) {
        try {
            /**
             * 设置入参
             * float* datt, const float* att, int B, int T, int C, float scale

             */
            int block_size = 256;
            softmaxBackwardParameters = Pointer.to(Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new int[]{C}), Pointer.to(new float[]{scale}));
            int grid_size = get_number_of_blocks(T, 4);
            int[] grids = new int[]{grid_size, B * NH, 1};
            checkCUDA(cuLaunchKernel(softmax_backward_unmask_2_function, grids[0], grids[1], grids[2],      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void softmax_kv_unmask_backward(Tensor datt, Tensor att, int B, int T, int T2, int NH, float scale) {
        try {
            /**
             * 设置入参
             * float* datt, const float* att, int B, int T, int C, float scale

             */
            int block_size = 256;
            softmaxBackwardParameters = Pointer.to(Pointer.to(datt.getGpuData()), Pointer.to(att.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{T}), Pointer.to(new int[]{T2}), Pointer.to(new float[]{scale}));
            int grid_size = get_number_of_blocks(T, 4);
            int[] grids = new int[]{grid_size, B * NH, 1};
            checkCUDA(cuLaunchKernel(softmax_backward_unmask_kv_function, grids[0], grids[1], grids[2],      // Grid dimension
                    block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    softmaxBackwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int get_number_of_blocks(int array_size, int block_size) {
        return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
}

