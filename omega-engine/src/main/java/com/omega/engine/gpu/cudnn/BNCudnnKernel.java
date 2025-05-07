package com.omega.engine.gpu.cudnn;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.network.RunModel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * BNCudnnKernel
 *
 * @author Administrator
 */
public class BNCudnnKernel extends BNBaseKernel {
    private BNType bnType;
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
    //	private Pointer diff;
    private CUfunction normalize_test_function;
    private Pointer normalize_test_Parameters;
    private int CAFFE_CUDA_NUM_THREADS = 1024;

    public BNCudnnKernel(BNType bnType, int C, int H, int W, Tensor runingMean, Tensor runingVar, CUDAManager cudaManager) {
        super(cudaManager);
        this.bnType = bnType;
        this.C = C;
        this.H = H;
        this.W = W;
        this.runingMean = runingMean;
        this.runingVar = runingVar;
        init();
    }

    public void init() {
        //		if(normalize_test_function == null) {
        //			normalize_test_function = CUDAModules.getLocalFunctionByModule("BNKernel3.cu", "normalize_test_kernel");
        //		}
        if (bnType == BNType.fully_bn) {
            mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
            mean = new Tensor(1, 1, 1, W, true);
            var = new Tensor(1, 1, 1, W, true);
        } else {
            mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
            mean = new Tensor(1, 1, 1, C, true);
            var = new Tensor(1, 1, 1, C, true);
        }
        normTensorDesc = new cudnnTensorDescriptor();
        dstTensorDesc = new cudnnTensorDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(normTensorDesc);
        JCudnn.cudnnCreateTensorDescriptor(dstTensorDesc);
        if (bnType == BNType.fully_bn) {
            JCudnn.cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, W, 1, 1);
        } else {
            JCudnn.cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);
        }
    }

    public void initForward(Tensor input) {
        if (input.number != this.N) {
            this.N = input.number;
            CudnnHandleManager.handle(JCudnn.cudnnDestroyTensorDescriptor(dstTensorDesc));
            CudnnHandleManager.handle(JCudnn.cudnnCreateTensorDescriptor(dstTensorDesc));
            if (bnType == BNType.fully_bn) {
                JCudnn.cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, 1, 1);
            } else {
                JCudnn.cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
            }
        }
    }

    public void forward(RunModel RUN_MODEL, Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        initForward(input);
        if (RUN_MODEL == RunModel.TRAIN) {
            //			JCudnn.cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
            //			JCudnn.cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
            //			JCudnn.cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardTraining(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, output.getGpuData(), normTensorDesc, gamma.getGpuData(), beta.getGpuData(), momentum, runingMean.getGpuData(), runingVar.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        } else {
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardInference(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, output.getGpuData(), normTensorDesc, gamma.getGpuData(), beta.getGpuData(), runingMean.getGpuData(), runingVar.getGpuData(), eps));
            //			normalize_test(input, gamma, beta, output);
        }
    }

    public void forward(RunModel RUN_MODEL, Tensor gamma, Tensor beta, Tensor input, Tensor output, int batch, int step) {
        initForward(input);
        if (RUN_MODEL == RunModel.TRAIN) {
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardTraining(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, dstTensorDesc, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), dstTensorDesc, output.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), normTensorDesc, gamma.getGpuData(), beta.getGpuData(), momentum, runingMean.getGpuData(), runingVar.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        } else {
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardInference(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, dstTensorDesc, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), dstTensorDesc, output.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), normTensorDesc, gamma.getGpuData(), beta.getGpuData(), runingMean.getGpuData(), runingVar.getGpuData(), eps));
        }
    }

    public void normalize_test(Tensor input, Tensor gama, Tensor beta, Tensor output) {
        try {
            /**
             * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial

             */
            normalize_test_Parameters = Pointer.to(Pointer.to(new int[]{N * C * H * W}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(runingMean.getGpuData()), Pointer.to(runingVar.getGpuData()), Pointer.to(gama.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H * W}), Pointer.to(new float[]{(float) eps}));
            cuLaunchKernel(normalize_test_function, this.CAFFE_GET_BLOCKS(N * C * H * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    normalize_test_Parameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        if (dgamma == null) {
            System.err.println("in");
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, delta.getGpuData(), dstTensorDesc, diff.getGpuData(), normTensorDesc, gamma.getGpuData(), null, null, eps, mean.getGpuData(), var.getGpuData()));
        } else {
            CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, delta.getGpuData(), dstTensorDesc, diff.getGpuData(), normTensorDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta, int batch, int step) {
        CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode, alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), dstTensorDesc, delta.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), dstTensorDesc, diff.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT), normTensorDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
    }
}



