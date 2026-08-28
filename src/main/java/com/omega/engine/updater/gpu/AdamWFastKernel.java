package com.omega.engine.updater.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

public class AdamWFastKernel extends BaseKernel {

    private static final int CAFFE_CUDA_NUM_THREADS = 256;

    private CUfunction adamwMultiTensorFunction;

    private CUdeviceptr weightsGpu;
    private CUdeviceptr gradsGpu;
    private CUdeviceptr msGpu;
    private CUdeviceptr vsGpu;
    private CUdeviceptr offsetsGpu;

    private int cachedCount = -1;
    private int cachedHash = 0;

    public AdamWFastKernel(CUDAManager cudaManager) {
        super(cudaManager);
        initFunction();
    }

    public void initFunction() {
        try {
            if (adamwMultiTensorFunction == null) {
                adamwMultiTensorFunction = getCudaManager().getLocalFunctionByModule("updaterFast.cu", "adamw_multi_tensor_kernel");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void update(Tensor[] weights, Tensor[] grads, Tensor[] ms, Tensor[] vs, int[] lengths, int totalLength, int tensorCount, float lr,
            float beta1, float beta2, float beta1Correction, float beta2Correction, float eps, float weightDecay) {
        if (tensorCount <= 0 || totalLength <= 0) {
            return;
        }
        uploadIfNeeded(weights, grads, ms, vs, lengths, tensorCount);
        Pointer kernelParameters = Pointer.to(
                Pointer.to(weightsGpu),
                Pointer.to(gradsGpu),
                Pointer.to(msGpu),
                Pointer.to(vsGpu),
                Pointer.to(offsetsGpu),
                Pointer.to(new int[] {tensorCount}),
                Pointer.to(new int[] {totalLength}),
                Pointer.to(new float[] {lr}),
                Pointer.to(new float[] {beta1}),
                Pointer.to(new float[] {beta2}),
                Pointer.to(new float[] {beta1Correction}),
                Pointer.to(new float[] {beta2Correction}),
                Pointer.to(new float[] {eps}),
                Pointer.to(new float[] {weightDecay})
        );
        cuLaunchKernel(adamwMultiTensorFunction, CAFFE_GET_BLOCKS(totalLength), 1, 1,
                CAFFE_CUDA_NUM_THREADS, 1, 1, 0, null, kernelParameters, null);
    }

    private void uploadIfNeeded(Tensor[] weights, Tensor[] grads, Tensor[] ms, Tensor[] vs, int[] lengths, int tensorCount) {
        int hash = hashPointers(weights, grads, ms, vs, lengths, tensorCount);
        if (tensorCount == cachedCount && hash == cachedHash) {
            return;
        }
        ensurePointerBuffers(tensorCount);

        Pointer[] weightPointers = new Pointer[tensorCount];
        Pointer[] gradPointers = new Pointer[tensorCount];
        Pointer[] mPointers = new Pointer[tensorCount];
        Pointer[] vPointers = new Pointer[tensorCount];
        int[] offsets = new int[tensorCount + 1];
        for (int i = 0; i < tensorCount; i++) {
            weightPointers[i] = weights[i].getGpuData();
            gradPointers[i] = grads[i].getGpuData();
            mPointers[i] = ms[i].getGpuData();
            vPointers[i] = vs[i].getGpuData();
            offsets[i + 1] = offsets[i] + lengths[i];
        }

        cuMemcpyHtoD(weightsGpu, Pointer.to(weightPointers), tensorCount * (long) Sizeof.POINTER);
        cuMemcpyHtoD(gradsGpu, Pointer.to(gradPointers), tensorCount * (long) Sizeof.POINTER);
        cuMemcpyHtoD(msGpu, Pointer.to(mPointers), tensorCount * (long) Sizeof.POINTER);
        cuMemcpyHtoD(vsGpu, Pointer.to(vPointers), tensorCount * (long) Sizeof.POINTER);
        cuMemcpyHtoD(offsetsGpu, Pointer.to(offsets), (tensorCount + 1L) * Sizeof.INT);

        cachedCount = tensorCount;
        cachedHash = hash;
    }

    private void ensurePointerBuffers(int tensorCount) {
        if (weightsGpu != null && tensorCount <= cachedCount) {
            return;
        }
        weightsGpu = new CUdeviceptr();
        gradsGpu = new CUdeviceptr();
        msGpu = new CUdeviceptr();
        vsGpu = new CUdeviceptr();
        offsetsGpu = new CUdeviceptr();
        cuMemAlloc(weightsGpu, tensorCount * (long) Sizeof.POINTER);
        cuMemAlloc(gradsGpu, tensorCount * (long) Sizeof.POINTER);
        cuMemAlloc(msGpu, tensorCount * (long) Sizeof.POINTER);
        cuMemAlloc(vsGpu, tensorCount * (long) Sizeof.POINTER);
        cuMemAlloc(offsetsGpu, (tensorCount + 1L) * Sizeof.INT);
    }

    private int hashPointers(Tensor[] weights, Tensor[] grads, Tensor[] ms, Tensor[] vs, int[] lengths, int tensorCount) {
        int hash = tensorCount;
        for (int i = 0; i < tensorCount; i++) {
            hash = 31 * hash + System.identityHashCode(weights[i].getGpuData());
            hash = 31 * hash + System.identityHashCode(grads[i].getGpuData());
            hash = 31 * hash + System.identityHashCode(ms[i].getGpuData());
            hash = 31 * hash + System.identityHashCode(vs[i].getGpuData());
            hash = 31 * hash + lengths[i];
        }
        return hash;
    }

    public int CAFFE_GET_BLOCKS(int n) {
        return (n + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
}
