package com.omega.engine.gpu.cudnn;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class SoftmaxCudnnKernel extends BaseKernel {
    private int C;
    private int H;
    private int W;
    private Pointer alpha_P = Pointer.to(new float[]{1});
    private Pointer beta_P = Pointer.to(new float[]{0});
    private cudnnTensorDescriptor xDesc;
    private cudnnTensorDescriptor diffDesc;
    private cudnnTensorDescriptor yDesc;

    public SoftmaxCudnnKernel(int C, int H, int W, CUDAManager cudaManager) {
        super(cudaManager);
        this.C = C;
        this.H = H;
        this.W = W;
        xDesc = new cudnnTensorDescriptor();
        diffDesc = new cudnnTensorDescriptor();
        yDesc = new cudnnTensorDescriptor();
        handle(JCudnn.cudnnCreateTensorDescriptor(xDesc));
        handle(JCudnn.cudnnCreateTensorDescriptor(diffDesc));
        handle(JCudnn.cudnnCreateTensorDescriptor(yDesc));
    }

    /**
     * Handle.
     *
     * @param returnCode the return run
     */
    public static void handle(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
            throw new RuntimeException(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
        }
    }

    public static String checkError(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            return jcuda.jcudnn.cudnnStatus.stringFor(returnCode);
        } else {
            return "success";
        }
    }

    public void init(int number) {
        if (this.N != number) {
            this.N = number;
            //			System.out.println(this.N+":"+C+":"+H+":"+W);
            handle(JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
            handle(JCudnn.cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
            handle(JCudnn.cudnnSetTensor4dDescriptor(diffDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
        }
    }

    public void softmax(Tensor input, Tensor output) {
        init(input.number);
        handle(JCudnn.cudnnSoftmaxForward(CudnnHandleManager.getHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha_P, xDesc, input.getGpuData(), beta_P, yDesc, output.getGpuData()));
    }

    public void softmax(Tensor input, Tensor output, int number) {
        init(number);
        handle(JCudnn.cudnnSoftmaxForward(CudnnHandleManager.getHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha_P, xDesc, input.getGpuData(), beta_P, yDesc, output.getGpuData()));
    }

    public void softmax_backward(Tensor output, Tensor delta, Tensor diff) {
        handle(JCudnn.cudnnSoftmaxBackward(CudnnHandleManager.getHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha_P, xDesc, output.getGpuData(), diffDesc, delta.getGpuData(), beta_P, yDesc, diff.getGpuData()));
    }
}

