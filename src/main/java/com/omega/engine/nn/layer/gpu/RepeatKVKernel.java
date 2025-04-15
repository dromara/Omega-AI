package com.omega.engine.nn.layer.gpu;

import com.omega.common.config.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * RepeatKVKernel
 */
public class RepeatKVKernel extends BaseKernel {
    /**
     * 向前方法
     */
    private CUfunction forward_once_function;
    private CUfunction forward_function;
    /**
     * 反向传播方法
     */
    private CUfunction backward_once_function;
    private CUfunction backward_function;
    /**
     * 前向方法参数
     */
    private Pointer forwardParameters;
    private Pointer backwardParameters;

    public RepeatKVKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String[] args) {
        try {
            int N = 2;
            int T = 3;
            int HN = 2;
            int W = 4;
            int nRep = 2;
            float[] data = RandomUtils.order(N * T * HN * W, 0.01f, 0.01f);
            Tensor k = new Tensor(N, T, HN, W, data, true);
            Tensor v = new Tensor(N, T, HN, W, data, true);
            Tensor ok = new Tensor(N, T, HN * nRep, W, true);
            Tensor ov = new Tensor(N, T, HN * nRep, W, true);
            Tensor deltaK = new Tensor(N, T, HN * nRep, W, RandomUtils.order(N * T * HN * nRep * W, 0.01f, 0.01f), true);
            Tensor deltaV = new Tensor(N, T, HN * nRep, W, RandomUtils.order(N * T * HN * nRep * W, 0.01f, 0.01f), true);
            Tensor dk = new Tensor(N, T, HN, W, true);
            Tensor dv = new Tensor(N, T, HN, W, true);
            CUDAManager cudaManager = new CUDAManager(0);
            RepeatKVKernel kernel = new RepeatKVKernel(cudaManager);
            for (int i = 0; i < 10; i++) {
                kernel.forward(k, v, ok, ov, nRep);
                ok.showDM();
                ov.showDM();
                kernel.backward(deltaK, deltaV, dk, dv, nRep);
                dk.showDM();
                dv.showDM();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void initFunction() {
        try {
            if (forward_once_function == null) {
                forward_once_function = getCudaManager().getLocalFunctionByModule("RepeatKVKernel.cu", "repeat_once_forward");
            }
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("RepeatKVKernel.cu", "repeat_kv_forward");
            }
            if (backward_once_function == null) {
                backward_once_function = getCudaManager().getLocalFunctionByModule("RepeatKVKernel.cu", "repeat_once_backward");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("RepeatKVKernel.cu", "repeat_kv_backward");
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

    public void forward(Tensor input, Tensor output, int nRep) {
        try {
            /**
             * float *k_out, float *v_out, const float *k, const float *v,int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim

             */
            forwardParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(new int[]{input.number}), //batchSize
                    Pointer.to(new int[]{input.channel}),  //Time
                    Pointer.to(new int[]{input.height}),  //kv_head_num
                    Pointer.to(new int[]{nRep}),  //query_per_kv_head = q_head_num / kv_head_num
                    Pointer.to(new int[]{input.width}) //head_dim
            );
            int[] block_nums = new int[]{input.number, input.channel, input.height};
            int[] block_dims = new int[]{input.width, 1, 1};
            checkCUDA(cuLaunchKernel(forward_once_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Tensor k, Tensor v, Tensor ok, Tensor ov, int nRep) {
        try {
            /**
             * float *k_out, float *v_out, const float *k, const float *v,int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim

             */
            forwardParameters = Pointer.to(Pointer.to(ok.getGpuData()), Pointer.to(ov.getGpuData()), Pointer.to(k.getGpuData()), Pointer.to(v.getGpuData()), Pointer.to(new int[]{k.number}), //batchSize
                    Pointer.to(new int[]{k.channel}),  //Time
                    Pointer.to(new int[]{k.height}),  //kv_head_num
                    Pointer.to(new int[]{nRep}),  //query_per_kv_head = q_head_num / kv_head_num
                    Pointer.to(new int[]{k.width}) //head_dim
            );
            int[] block_nums = new int[]{k.number, k.channel, k.height};
            int[] block_dims = new int[]{k.width, 1, 1};
            checkCUDA(cuLaunchKernel(forward_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor detla, Tensor diff, int nRep) {
        try {
            /**
             * float *dk, float *dv, const float *dk_rep, const float *dv_rep,int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim

             */
            backwardParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(detla.getGpuData()), Pointer.to(new int[]{diff.number}), //batchSize
                    Pointer.to(new int[]{diff.channel}),  //Time
                    Pointer.to(new int[]{diff.height}),  //kv_head_num
                    Pointer.to(new int[]{nRep}),  //query_per_kv_head = q_head_num / kv_head_num
                    Pointer.to(new int[]{diff.width}) //head_dim
            );
            int[] block_nums = new int[]{diff.number, diff.channel, diff.height};
            int[] block_dims = new int[]{diff.width, 1, 1};
            checkCUDA(cuLaunchKernel(backward_once_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor dRepK, Tensor dRepV, Tensor diffK, Tensor diffV, int nRep) {
        try {
            /**
             * float *dk, float *dv, const float *dk_rep, const float *dv_rep,int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim

             */
            backwardParameters = Pointer.to(Pointer.to(diffK.getGpuData()), Pointer.to(diffV.getGpuData()), Pointer.to(dRepK.getGpuData()), Pointer.to(dRepV.getGpuData()), Pointer.to(new int[]{diffK.number}), //batchSize
                    Pointer.to(new int[]{diffK.channel}),  //Time
                    Pointer.to(new int[]{diffK.height}),  //kv_head_num
                    Pointer.to(new int[]{nRep}),  //query_per_kv_head = q_head_num / kv_head_num
                    Pointer.to(new int[]{diffK.width}) //head_dim
            );
            int[] block_nums = new int[]{diffK.number, diffK.channel, diffK.height};
            int[] block_dims = new int[]{diffK.width, 1, 1};
            checkCUDA(cuLaunchKernel(backward_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    public void showDM(String id, CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
        System.out.println(id + ":" + JsonUtils.toJson(data));
    }
}

