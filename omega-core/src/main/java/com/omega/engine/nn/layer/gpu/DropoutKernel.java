package com.omega.engine.nn.layer.gpu;

import com.omega.common.tensor.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class DropoutKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction back_function;
    private CUfunction dropout_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private int BLOCK = 512;
    private Pointer kernelParameters;
    private Pointer kernelBackParameters;
    private Pointer dropoutKernelParameters;
    private float prob = 0.0f;
    private float scale = 1.0f;

    public DropoutKernel(float prob, float scale, CUDAManager cudaManager) {
        super(cudaManager);
        this.prob = prob;
        this.scale = scale;
        init();
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
                function = getCudaManager().getLocalFunctionByModule("DropoutKernel.cu", "forward_kernel");
            }
            if (back_function == null) {
                back_function = getCudaManager().getLocalFunctionByModule("DropoutKernel.cu", "backward_kernel");
            }
            if (dropout_function == null) {
                dropout_function = getCudaManager().getLocalFunctionByModule("DropoutKernel.cu", "dropout_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public int get_number_of_blocks(int array_size, int block_size) {
        return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
    }

    public void forward(Tensor input, Tensor rand) {
        try {
            //			if(kernelParameters == null || input.number != this.N){
            /**
             * 设置入参
             * float *input, int size, float *rand, float prob, float scale

             */
            kernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(new int[]{input.getDataLength()}), Pointer.to(rand.getGpuData()), Pointer.to(new float[]{prob}), Pointer.to(new float[]{scale}));
            this.N = input.number;
            //			}
            cuLaunchKernel(function, this.get_number_of_blocks(input.getDataLength(), BLOCK), 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor delta, Tensor rand) {
        try {
            if (kernelBackParameters == null || delta.number != this.N) {
                /**
                 * 设置入参
                 * float *input, int size, float *rand, float prob, float scale

                 */
                kernelBackParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.getDataLength()}), Pointer.to(rand.getGpuData()), Pointer.to(new float[]{prob}), Pointer.to(new float[]{scale}));
                this.N = delta.number;
            }
            cuLaunchKernel(back_function, this.get_number_of_blocks(delta.getDataLength(), BLOCK), 1, 1,      // Grid dimension
                    BLOCK, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelBackParameters, null // Kernel- and extra parameters
            );
            //			delta.showDMByNumber(0);
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void dropout(Tensor input, Tensor output, Tensor rand) {
        try {
            //			if(dropoutKernelParameters == null || input.number != this.N){
            /**
             * 设置入参
             * float *input, float *output, int size, float *rand, float prob, float scale

             */
            dropoutKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(rand.getGpuData()), Pointer.to(new int[]{input.getDataLength()}), Pointer.to(new float[]{prob}), Pointer.to(new float[]{scale}));
            this.N = input.number;
            //		        rand.showDM(0);
            //			}
            //			int[] grid = cuda_gridsize(input.getDataLength());
            cuLaunchKernel(dropout_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null,               // Shared memory size and stream
                    dropoutKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] cuda_gridsize(int n) {
        int k = (n - 1) / BLOCK + 1;
        int x = k;
        int y = 1;
        if (x > 65535) {
            x = (int) Math.ceil(Math.sqrt(k));
            y = (n - 1) / (x * BLOCK) + 1;
        }
        //dim3 d = { (unsigned int)x, (unsigned int)y, 1 };
        int[] d = new int[3];
        d[0] = x;
        d[1] = y;
        d[2] = 1;
        //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
        return d;
    }
}

