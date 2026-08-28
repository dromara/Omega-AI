package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class DiTQKVFusedKernel extends BaseKernel {

    private static final int CAFFE_CUDA_NUM_THREADS = 1024;

    private CUfunction splitPermuteFunction;
    private CUfunction mergeUnpermuteFunction;

    public DiTQKVFusedKernel(CUDAManager cudaManager) {
        super(cudaManager);
        initFunction();
    }

    private void initFunction() {
        if (splitPermuteFunction == null) {
            splitPermuteFunction = getCudaManager().getLocalFunctionByModule("DiTQKVFusedKernel.cu", "qkv_split_permute_kernel");
        }
        if (mergeUnpermuteFunction == null) {
            mergeUnpermuteFunction = getCudaManager().getLocalFunctionByModule("DiTQKVFusedKernel.cu", "qkv_merge_unpermute_kernel");
        }
    }

    public void splitPermute(Tensor qkv, Tensor q, Tensor k, Tensor v, int batchSize, int time, int headNum, int dk) {
        initFunction();
        int len = batchSize * headNum * time * dk;
        Pointer params = Pointer.to(
                Pointer.to(qkv.getGpuData()),
                Pointer.to(q.getGpuData()),
                Pointer.to(k.getGpuData()),
                Pointer.to(v.getGpuData()),
                Pointer.to(new int[]{batchSize}),
                Pointer.to(new int[]{time}),
                Pointer.to(new int[]{headNum}),
                Pointer.to(new int[]{dk})
        );
        checkCUDA(cuLaunchKernel(splitPermuteFunction, CAFFE_GET_BLOCKS(len), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void mergeUnpermute(Tensor dq, Tensor dkey, Tensor dv, Tensor dqkv, int batchSize, int time, int headNum, int dk) {
        initFunction();
        int len = batchSize * headNum * time * dk;
        Pointer params = Pointer.to(
                Pointer.to(dq.getGpuData()),
                Pointer.to(dkey.getGpuData()),
                Pointer.to(dv.getGpuData()),
                Pointer.to(dqkv.getGpuData()),
                Pointer.to(new int[]{batchSize}),
                Pointer.to(new int[]{time}),
                Pointer.to(new int[]{headNum}),
                Pointer.to(new int[]{dk})
        );
        checkCUDA(cuLaunchKernel(mergeUnpermuteFunction, CAFFE_GET_BLOCKS(len), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }
}
