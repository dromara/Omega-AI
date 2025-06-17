package com.omega.engine.gpu.cudnn;

import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.layer.gpu.PoolingBaseKernel;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * PoolingCudnnKernel
 *
 * @author Administrator
 */
public class PoolingCudnnKernel extends PoolingBaseKernel {
    private int C;
    private int H;
    private int W;
    private int oc;
    private int oh;
    private int ow;
    private int pWidth;
    private int pHeight;
    private int padding = 0;
    private int stride = 1;
    private Pointer alpha_P = Pointer.to(new float[]{1.0f});
    private Pointer beta_P = Pointer.to(new float[]{0.0f});
    private cudnnTensorDescriptor xDesc;
    private cudnnPoolingDescriptor poolingDesc;
    private cudnnTensorDescriptor yDesc;

    public PoolingCudnnKernel(PoolingType poolingType, int C, int H, int W, int oh, int ow, int pWidth, int pHeight, int stride, int padding, CUDAManager cudaManager) {
        super(cudaManager);
        this.C = C;
        this.H = H;
        this.W = W;
        this.oc = C;
        this.oh = oh;
        this.ow = ow;
        this.pWidth = pWidth;
        this.pHeight = pHeight;
        this.padding = padding / 2;
        this.stride = stride;
        xDesc = new cudnnTensorDescriptor();
        poolingDesc = new cudnnPoolingDescriptor();
        yDesc = new cudnnTensorDescriptor();
        handle(JCudnn.cudnnCreateTensorDescriptor(xDesc));
        handle(JCudnn.cudnnCreatePoolingDescriptor(poolingDesc));
        handle(JCudnn.cudnnCreateTensorDescriptor(yDesc));
        int mode = 0;
        switch (poolingType) {
            case MAX_POOLING:
                mode = CUDNN_POOLING_MAX;
                break;
            case AVG_POOLING:
                mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                break;
            default:
                break;
        }
        handle(JCudnn.cudnnSetPooling2dDescriptor(poolingDesc, mode, CUDNN_PROPAGATE_NAN, this.pHeight, this.pWidth, this.padding, this.padding, this.stride, this.stride));
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
            handle(JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
            handle(JCudnn.cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, this.oc, this.oh, this.ow));
        }
    }

    public void forward(Tensor input, Tensor output) {
        this.init(input.getShape()[0]);
        handle(JCudnn.cudnnPoolingForward(CudnnHandleManager.getHandle(), poolingDesc, alpha_P, xDesc, input.getGpuData(), beta_P, yDesc, output.getGpuData()));
    }

    @Override
    public void backward(Tensor input, Tensor output, Tensor delta, Tensor diff) {
        handle(JCudnn.cudnnPoolingBackward(CudnnHandleManager.getHandle(), poolingDesc, alpha_P, yDesc, output.getGpuData(), yDesc, delta.getGpuData(), xDesc, input.getGpuData(), beta_P, xDesc, diff.getGpuData()));
    }
}



