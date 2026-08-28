package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class DiTAdaLNKernel extends BaseKernel {

    private static final int CAFFE_CUDA_NUM_THREADS = 1024;

    private CUfunction split6Function;
    private CUfunction set6Function;

    public DiTAdaLNKernel(CUDAManager cudaManager) {
        super(cudaManager);
        initFunction();
    }

    private void initFunction() {
        if (split6Function == null) {
            split6Function = getCudaManager().getLocalFunctionByModule("DiTAdaLNKernel.cu", "adaln_split6_kernel");
        }
        if (set6Function == null) {
            set6Function = getCudaManager().getLocalFunctionByModule("DiTAdaLNKernel.cu", "adaln_set6_kernel");
        }
    }

    public void split6(Tensor input, Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa, Tensor shiftMlp, Tensor scaleMlp, Tensor gateMlp, int batchSize, int embedDim) {
        initFunction();
        int len = batchSize * embedDim;
        Pointer params = Pointer.to(
                Pointer.to(input.getGpuData()),
                Pointer.to(shiftMsa.getGpuData()),
                Pointer.to(scaleMsa.getGpuData()),
                Pointer.to(gateMsa.getGpuData()),
                Pointer.to(shiftMlp.getGpuData()),
                Pointer.to(scaleMlp.getGpuData()),
                Pointer.to(gateMlp.getGpuData()),
                Pointer.to(new int[]{batchSize}),
                Pointer.to(new int[]{embedDim})
        );
        checkCUDA(cuLaunchKernel(split6Function, CAFFE_GET_BLOCKS(len), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void set6(Tensor output, Tensor dShiftMsa, Tensor dScaleMsa, Tensor dGateMsa, Tensor dShiftMlp, Tensor dScaleMlp, Tensor dGateMlp, int batchSize, int embedDim) {
        initFunction();
        int len = batchSize * embedDim;
        Pointer params = Pointer.to(
                Pointer.to(output.getGpuData()),
                Pointer.to(dShiftMsa.getGpuData()),
                Pointer.to(dScaleMsa.getGpuData()),
                Pointer.to(dGateMsa.getGpuData()),
                Pointer.to(dShiftMlp.getGpuData()),
                Pointer.to(dScaleMlp.getGpuData()),
                Pointer.to(dGateMlp.getGpuData()),
                Pointer.to(new int[]{batchSize}),
                Pointer.to(new int[]{embedDim})
        );
        checkCUDA(cuLaunchKernel(set6Function, CAFFE_GET_BLOCKS(len), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }
}
