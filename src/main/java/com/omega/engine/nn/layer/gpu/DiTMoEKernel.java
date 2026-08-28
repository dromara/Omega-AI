package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class DiTMoEKernel extends BaseKernel {

    private static final int CAFFE_CUDA_NUM_THREADS = 1024;

    private CUfunction routeTop1Function;
    private CUfunction auxLossFunction;
    private CUfunction combineForwardFunction;
    private CUfunction dispatchForwardFunction;
    private CUfunction combineSparseForwardFunction;
    private CUfunction expertDeltaFunction;
    private CUfunction sparseExpertDeltaFunction;
    private CUfunction scatterAddDiffFunction;
    private CUfunction swigluForwardFunction;
    private CUfunction swigluBackwardFunction;
    private CUfunction addBiasFunction;
    private CUfunction biasBackwardFunction;
    private CUfunction gateBackwardFunction;

    public DiTMoEKernel(CUDAManager cudaManager) {
        super(cudaManager);
        initFunction();
    }

    private void initFunction() {
        if (routeTop1Function == null) {
            routeTop1Function = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_route_top1_kernel");
        }
        if (auxLossFunction == null) {
            auxLossFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_aux_loss_kernel");
        }
        if (combineForwardFunction == null) {
            combineForwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_combine_forward_kernel");
        }
        if (dispatchForwardFunction == null) {
            dispatchForwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_dispatch_forward_kernel");
        }
        if (combineSparseForwardFunction == null) {
            combineSparseForwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_combine_sparse_forward_kernel");
        }
        if (expertDeltaFunction == null) {
            expertDeltaFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_expert_delta_kernel");
        }
        if (sparseExpertDeltaFunction == null) {
            sparseExpertDeltaFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_sparse_expert_delta_kernel");
        }
        if (scatterAddDiffFunction == null) {
            scatterAddDiffFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_scatter_add_diff_kernel");
        }
        if (swigluForwardFunction == null) {
            swigluForwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_swiglu_forward_kernel");
        }
        if (swigluBackwardFunction == null) {
            swigluBackwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_swiglu_backward_kernel");
        }
        if (addBiasFunction == null) {
            addBiasFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_add_bias_kernel");
        }
        if (biasBackwardFunction == null) {
            biasBackwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_bias_backward_kernel");
        }
        if (gateBackwardFunction == null) {
            gateBackwardFunction = getCudaManager().getLocalFunctionByModule("DiTMoEKernel.cu", "moe_gate_backward_kernel");
        }
    }

    public void routeTop1(Tensor logits, Tensor probs, Tensor topIdx, Tensor topWeight, Tensor expertCounts, int N, int E) {
        routeTop1(logits, probs, topIdx, null, topWeight, expertCounts, N, E);
    }

    public void routeTop1(Tensor logits, Tensor probs, Tensor topIdx, Tensor topSlot, Tensor topWeight, Tensor expertCounts, int N, int E) {
        initFunction();
        expertCounts.clearGPU();
        Tensor slot = topSlot == null ? topWeight : topSlot;
        Pointer params = Pointer.to(
                Pointer.to(logits.getGpuData()),
                Pointer.to(probs.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(slot.getGpuData()),
                Pointer.to(topWeight.getGpuData()),
                Pointer.to(expertCounts.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{E})
        );
        checkCUDA(cuLaunchKernel(routeTop1Function, CAFFE_GET_BLOCKS(N), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void auxLoss(Tensor probs, Tensor expertCounts, Tensor auxLoss, int N, int E, float coef) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(probs.getGpuData()),
                Pointer.to(expertCounts.getGpuData()),
                Pointer.to(auxLoss.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{E}),
                Pointer.to(new float[]{coef})
        );
        checkCUDA(cuLaunchKernel(auxLossFunction, 1, 1, 1,
                1, 1, 1, 0, null, params, null));
    }

    public void combineForward(Tensor expertOut, Tensor out, Tensor topIdx, Tensor topWeight, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(expertOut.getGpuData()),
                Pointer.to(out.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topWeight.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(combineForwardFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void dispatchForward(Tensor input, Tensor expertInput, Tensor topIdx, Tensor topSlot, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(input.getGpuData()),
                Pointer.to(expertInput.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topSlot.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(dispatchForwardFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void combineSparseForward(Tensor expertOut, Tensor out, Tensor topIdx, Tensor topSlot, Tensor topWeight, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(expertOut.getGpuData()),
                Pointer.to(out.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topSlot.getGpuData()),
                Pointer.to(topWeight.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(combineSparseForwardFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void expertDelta(Tensor delta, Tensor expertOut, Tensor expertDelta, Tensor topWeightGrad, Tensor topIdx, Tensor topWeight, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(delta.getGpuData()),
                Pointer.to(expertOut.getGpuData()),
                Pointer.to(expertDelta.getGpuData()),
                Pointer.to(topWeightGrad.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topWeight.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(expertDeltaFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void sparseExpertDelta(Tensor delta, Tensor expertOut, Tensor expertDelta, Tensor topWeightGrad, Tensor topIdx, Tensor topSlot, Tensor topWeight, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(delta.getGpuData()),
                Pointer.to(expertOut.getGpuData()),
                Pointer.to(expertDelta.getGpuData()),
                Pointer.to(topWeightGrad.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topSlot.getGpuData()),
                Pointer.to(topWeight.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(sparseExpertDeltaFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void scatterAddDiff(Tensor expertDiff, Tensor diff, Tensor topIdx, Tensor topSlot, int expertId, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(expertDiff.getGpuData()),
                Pointer.to(diff.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topSlot.getGpuData()),
                Pointer.to(new int[]{expertId}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(scatterAddDiffFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void swigluForward(Tensor w12Out, Tensor w1, Tensor w2, Tensor act, Tensor wt, int N, int H) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(w12Out.getGpuData()),
                Pointer.to(w1.getGpuData()),
                Pointer.to(w2.getGpuData()),
                Pointer.to(act.getGpuData()),
                Pointer.to(wt.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{H})
        );
        checkCUDA(cuLaunchKernel(swigluForwardFunction, CAFFE_GET_BLOCKS(N * H), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void swigluBackward(Tensor dwt, Tensor w1, Tensor w2, Tensor act, Tensor w12Delta, int N, int H) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(dwt.getGpuData()),
                Pointer.to(w1.getGpuData()),
                Pointer.to(w2.getGpuData()),
                Pointer.to(act.getGpuData()),
                Pointer.to(w12Delta.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{H})
        );
        checkCUDA(cuLaunchKernel(swigluBackwardFunction, CAFFE_GET_BLOCKS(N * H), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void addBias(Tensor output, Tensor bias, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(output.getGpuData()),
                Pointer.to(bias.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(addBiasFunction, CAFFE_GET_BLOCKS(N * C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void biasBackward(Tensor delta, Tensor diffBias, int N, int C) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(delta.getGpuData()),
                Pointer.to(diffBias.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{C})
        );
        checkCUDA(cuLaunchKernel(biasBackwardFunction, CAFFE_GET_BLOCKS(C), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }

    public void gateBackward(Tensor probs, Tensor topIdx, Tensor topWeightGrad, Tensor expertCounts, Tensor gateDelta, int N, int E, float coef) {
        initFunction();
        Pointer params = Pointer.to(
                Pointer.to(probs.getGpuData()),
                Pointer.to(topIdx.getGpuData()),
                Pointer.to(topWeightGrad.getGpuData()),
                Pointer.to(expertCounts.getGpuData()),
                Pointer.to(gateDelta.getGpuData()),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{E}),
                Pointer.to(new float[]{coef})
        );
        checkCUDA(cuLaunchKernel(gateBackwardFunction, CAFFE_GET_BLOCKS(N * E), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, params, null));
    }
}
