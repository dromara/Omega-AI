package com.omega.engine.nn.layer.opensora.wfvae;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class WFKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction function_expend;
    private CUfunction function_4_expend;
    private CUfunction function_append;
    private CUfunction function_back;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public WFKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 8;
        float[] x1 = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float[] x2 = MatrixOperation.multiplication(x1, 2);
        float[] x3 = MatrixOperation.multiplication(x1, 3);
        float[] x4 = MatrixOperation.multiplication(x1, 4);
        float[] x5 = MatrixOperation.multiplication(x1, 5);
        float[] x6 = MatrixOperation.multiplication(x1, 6);
        float[] x7 = MatrixOperation.multiplication(x1, 7);
        float[] x8 = MatrixOperation.multiplication(x1, 8);
//        float[] bias1 = MatrixUtils.one(x1.length);
        Tensor input1 = new Tensor(N, C, H, W, x1, true);
        Tensor input2 = new Tensor(N, C, H, W, x2, true);
        Tensor input3 = new Tensor(N, C, H, W, x3, true);
        Tensor input4 = new Tensor(N, C, H, W, x4, true);
        Tensor input5 = new Tensor(N, C, H, W, x5, true);
        Tensor input6 = new Tensor(N, C, H, W, x6, true);
        Tensor input7 = new Tensor(N, C, H, W, x7, true);
        Tensor input8 = new Tensor(N, C, H, W, x8, true);
        Tensor output = new Tensor(N * 8, C, H, W, true);
        Tensor[] inputs = new Tensor[8];
        inputs[0] = input1;
        inputs[1] = input2;
        inputs[2] = input3;
        inputs[3] = input4;
        inputs[4] = input5;
        inputs[5] = input6;
        inputs[6] = input7;
        inputs[7] = input8;
        CUDAManager cudaManager = new CUDAManager(0);
        WFKernel k = new WFKernel(cudaManager);
        k.cat_number_expend(inputs, output, input1.getOnceSize());
        output.showDM();
        CUDAMemoryManager.free();
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("wf_vae.cu", "cat_number");
            }
            if (function_expend == null) {
            	function_expend = getCudaManager().getLocalFunctionByModule("wf_vae.cu", "cat_number_expend");
            }
            if (function_4_expend == null) {
            	function_4_expend = getCudaManager().getLocalFunctionByModule("wf_vae.cu", "cat_number_4_expend");
            }
            if (function_append == null) {
                function_append = getCudaManager().getLocalFunctionByModule("wf_vae.cu", "append");
            }
            if (function_back == null) {
                function_back = getCudaManager().getLocalFunctionByModule("wf_vae.cu", "cat_number");
            }

        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
    
    public void cat_number(Tensor[] input, Tensor output, int length) {
        try {
            /**
             * 设置入参
             * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
             */
        	Pointer[] list = new Pointer[input.length];
        	for(int i = 0;i<input.length;i++) {
        		list[i] = input[i].getGpuData();
        	}
            forwardKernelParameters = Pointer.to(Pointer.to(new int[] {output.dataLength / input.length}), Pointer.to(list), Pointer.to(new int[]{input.length}), Pointer.to(new int[]{length}), Pointer.to(output.getGpuData()));
            System.err.println(output.dataLength / input.length);
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength / input.length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cat_number_expend(Tensor[] input, Tensor output, int length) {
        try {
            /**
             * 设置入参
             *
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[] {output.dataLength / input.length}),
                    Pointer.to(input[0].getGpuData()),Pointer.to(input[1].getGpuData()),Pointer.to(input[2].getGpuData()),Pointer.to(input[3].getGpuData()),
                    Pointer.to(input[4].getGpuData()),Pointer.to(input[5].getGpuData()),Pointer.to(input[6].getGpuData()),Pointer.to(input[7].getGpuData()),
                    Pointer.to(new int[]{length}), Pointer.to(output.getGpuData()));

            cuLaunchKernel(function_expend, this.CAFFE_GET_BLOCKS(output.dataLength / input.length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cat_number_4_expend(Tensor[] input, Tensor output, int length) {
        try {
            /**
             * 设置入参
             * 
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[] {output.dataLength / input.length}),
            		Pointer.to(input[0].getGpuData()),Pointer.to(input[1].getGpuData()),Pointer.to(input[2].getGpuData()),Pointer.to(input[3].getGpuData()),
            		Pointer.to(new int[]{length}), Pointer.to(output.getGpuData()));

            cuLaunchKernel(function_4_expend, this.CAFFE_GET_BLOCKS(output.dataLength / input.length), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void append(Tensor input, Tensor output, int length, int offset) {
        try {
            /**
             * 设置入参
             *
             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()),
                    Pointer.to(output.getGpuData()),
                    Pointer.to(new int[]{length}), Pointer.to(new int[]{offset}));

            cuLaunchKernel(function_append, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

