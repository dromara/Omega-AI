package com.omega.engine.gpu.cudnn;

import com.omega.common.config.Tensor;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.network.RunModel;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * BNCudnnKernel
 *
 * @author Administrator
 */
public class InstanceNormalizationCudnnKernel extends BNBaseKernel {
    private int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
    private int C;
    private int H;
    private int W;
    private double eps = 1e-5;
    private double momentum = 0.01f;
    private Pointer alpha_P = Pointer.to(new float[]{1});
    private Pointer beta_P = Pointer.to(new float[]{0});
    private Tensor mean;
    private Tensor var;
    private cudnnTensorDescriptor normTensorDesc;
    private cudnnTensorDescriptor dstTensorDesc;
    private cudnnTensorDescriptor yTensorDesc;

    public InstanceNormalizationCudnnKernel(int C, int H, int W, CUDAManager cudaManager) {
        super(cudaManager);
        this.C = C;
        this.H = H;
        this.W = W;
        init();
    }

    public void init() {
        mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
        normTensorDesc = new cudnnTensorDescriptor();
        dstTensorDesc = new cudnnTensorDescriptor();
        yTensorDesc = new cudnnTensorDescriptor();
    }

    public void initForward(Tensor input) {
        if (input.number != this.N) {
            this.N = input.number;
            CudnnHandleManager.handle(JCudnn.cudnnDestroyTensorDescriptor(normTensorDesc));
            JCudnn.cudnnCreateTensorDescriptor(normTensorDesc);
            JCudnn.cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N * C, 1, 1);
            CudnnHandleManager.handle(JCudnn.cudnnDestroyTensorDescriptor(dstTensorDesc));
            CudnnHandleManager.handle(JCudnn.cudnnCreateTensorDescriptor(dstTensorDesc));
            JCudnn.cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N * C, H, W);
            mean = new Tensor(1, 1, 1, this.N * C, true);
            var = new Tensor(1, 1, 1, this.N * C, true);
        }
    }

    public void forward(RunModel RUN_MODEL, Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        initForward(input);
        CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardTraining(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, output.getGpuData(), normTensorDesc, gamma.getGpuData(), beta.getGpuData(), 0.1, null, null, eps, mean.getGpuData(), var.getGpuData()));
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, delta.getGpuData(), dstTensorDesc, diff.getGpuData(), normTensorDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
    }
}

	

