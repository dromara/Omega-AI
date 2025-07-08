package com.omega.engine.gpu.cudnn;

import static jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.layer.gpu.Conv3DBaseKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnConvolutionBwdDataAlgoPerf;
import jcuda.jcudnn.cudnnConvolutionBwdFilterAlgoPerf;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdAlgoPerf;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

public class Conv3DTransposeCudnnKernel extends Conv3DBaseKernel {
    private int C;
    private int F;
    private int H;
    private int W;
    private int ko;
    private int kf;
    private int kh;
    private int kw;
    private int on;
    private int oc;
    private int of;
    private int oh;
    private int ow;
    private int padding = 0;
    private int output_padding = 0;
    private int dilation = 1;
    private int stride = 1;
    private int convAlgorithm = -1;
    private int fw_algo;
    private int bkf_algo;
    private int bkd_algo;
    private Pointer alpha_P = Pointer.to(new float[]{1});
    private Pointer beta_P = Pointer.to(new float[]{0});
    private Network network;
    private cudnnTensorDescriptor xDesc;
    private cudnnFilterDescriptor kernelDesc;
    private cudnnTensorDescriptor yDesc;
    private cudnnConvolutionDescriptor convDesc;

    public Conv3DTransposeCudnnKernel(Network network, int C, int F, int H, int W, int ko, int kf, int kh, int kw, int s, int p, int d, int op, CUDAManager cudaManager) {
        super(cudaManager);
        this.network = network;
        this.C = C;
        this.F = F;
        this.H = H;
        this.W = W;
        this.ko = ko;
        this.kf = kf;
        this.kh = kh;
        this.kw = kw;
        this.stride = s;
        this.padding = p;
        this.output_padding = op;
        this.dilation = d;
        xDesc = new cudnnTensorDescriptor();
        kernelDesc = new cudnnFilterDescriptor();
        yDesc = new cudnnTensorDescriptor();
        convDesc = new cudnnConvolutionDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(xDesc);
        JCudnn.cudnnCreateFilterDescriptor(kernelDesc);
        JCudnn.cudnnCreateTensorDescriptor(yDesc);
        JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
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

    public int[] computeStride(int[] size) {
        int[] stride = new int[5];
        for (int i = 4; i >= 0; i--) {
            stride[i] = (i == 4) ? 1 : size[i + 1] * stride[i + 1];
        }
        return stride;
    }
    
    public void init(int number) {
        if (this.N != number) {
            this.N = number;
            int convDims = 3;
            int[] padA = {padding, padding, padding};
            int[] weight = {C, ko, kf, kh, kw};
            int[] upscaleA = {dilation, dilation, dilation};
            int[] tensorOuputDimA = {N, C, F, H, W};
//            System.err.println(JsonUtils.toJson(tensorOuputDimA));
            int[] strideA = computeStride(tensorOuputDimA);
//            System.err.println(JsonUtils.toJson(strideA));
            JCudnn.cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 5, tensorOuputDimA, strideA);
            JCudnn.cudnnSetFilterNdDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, weight);
            int[] filterStrideA = {stride, stride, stride};

            JCudnn.cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
 
            this.on = this.N;
            this.oc = this.ko;
            this.of = (F - 1) * stride - 2 * padA[0] + dilation * (this.kf - 1) + output_padding + 1;
            this.oh = (H - 1) * stride - 2 * padA[1] + dilation * (this.kh - 1) + output_padding + 1;
            this.ow = (W - 1) * stride - 2 * padA[2] + dilation * (this.kw - 1) + output_padding + 1;
            tensorOuputDimA[0] = on;
            tensorOuputDimA[1] = oc;
            tensorOuputDimA[2] = of;
            tensorOuputDimA[3] = oh;
            tensorOuputDimA[4] = ow;
            int[] out_strideA = computeStride(tensorOuputDimA);
//            System.err.println(JsonUtils.toJson(tensorOuputDimA));
            JCudnn.cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 5, tensorOuputDimA, out_strideA);
            this.fw_algo = getForwardAlgorithm(convAlgorithm, yDesc, kernelDesc, convDesc, xDesc);
            this.bkf_algo = getBKFGO(convDims, yDesc, xDesc, kernelDesc, convDesc);
            this.bkd_algo = getBKDGO(convDims, yDesc, xDesc, kernelDesc, convDesc);
            getWorkSpace();
        }
    }

    public void convTranspose(Tensor input, Tensor kernel, Tensor output) {
        this.init(input.number);
        handle(JCudnn.cudnnConvolutionBackwardData(CudnnHandleManager.getHandle(), alpha_P, kernelDesc, kernel.getGpuData(), xDesc, input.getGpuData(), convDesc, bkd_algo, this.network.workspace, this.network.workspaceSize, beta_P, yDesc, output.getGpuData()));
    }

    public void dw(Tensor input, Tensor delta, Tensor dKernel) {
        handle(JCudnn.cudnnConvolutionBackwardFilter(CudnnHandleManager.getHandle(), alpha_P, yDesc, delta.getGpuData(), xDesc, input.getGpuData(), convDesc, bkf_algo, this.network.workspace, this.network.workspaceSize, beta_P, kernelDesc, dKernel.getGpuData()));
    }

    public void dx(Tensor delta, Tensor kernel, Tensor diff) {
        handle(JCudnn.cudnnConvolutionForward(CudnnHandleManager.getHandle(), alpha_P, yDesc, delta.getGpuData(), kernelDesc, kernel.getGpuData(), convDesc, fw_algo, this.network.workspace, this.network.workspaceSize, beta_P, xDesc, diff.getGpuData()));
    }

    public int getBKDGO(int convAlgorithm, cudnnTensorDescriptor dxDesc, cudnnTensorDescriptor dyDesc, cudnnFilterDescriptor wDesc, cudnnConvolutionDescriptor convDesc) {
        int knum = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT * 2;
        int requestedAlgoCount = knum;
        int returnedAlgoCount = -1;
        int returnedAlgoCountArray[] = {knum};
        cudnnConvolutionBwdDataAlgoPerf results[] = new cudnnConvolutionBwdDataAlgoPerf[knum];
        //        System.out.println("Testing cudnnFindConvolutionBackwardDataAlgorithm ...");
        JCudnn.cudnnFindConvolutionBackwardDataAlgorithm(CudnnHandleManager.getHandle(), wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCountArray, results);
        returnedAlgoCount = returnedAlgoCountArray[0];
        //        for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex)
        //        {
        //       	 	String result = checkError(results[algoIndex].status);
        //            System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n",
        //                results[algoIndex].algo, results[algoIndex].time,
        //                (long)results[algoIndex].memory, "["+result+"]");
        //        }
        return results[0].algo;
    }

    public int getBKFGO(int convAlgorithm, cudnnTensorDescriptor xDesc, cudnnTensorDescriptor dyDesc, cudnnFilterDescriptor dwDesc, cudnnConvolutionDescriptor convDesc) {
        int requestedAlgoCount = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int returnedAlgoCountArray[] = {returnedAlgoCount};
        cudnnConvolutionBwdFilterAlgoPerf results[] = new cudnnConvolutionBwdFilterAlgoPerf[2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        //        System.out.println("Testing cudnnFindConvolutionBackwardFilterAlgorithm ...");
        JCudnn.cudnnFindConvolutionBackwardFilterAlgorithm(CudnnHandleManager.getHandle(), xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCountArray, results);
        returnedAlgoCount = returnedAlgoCountArray[0];
        //        for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
        //       	 	String result = checkError(results[algoIndex].status);
        //            System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n",
        //                results[algoIndex].algo, results[algoIndex].time,
        //                (long)results[algoIndex].memory, "["+result+"]");
        //        }
        return results[0].algo;
    }

    public int getForwardAlgorithm(int convAlgorithm, cudnnTensorDescriptor xDesc, cudnnFilterDescriptor wDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstDesc) {
        if (convAlgorithm < 0) {
            int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
            int returnedAlgoCount = -1;
            int returnedAlgoCountArray[] = {returnedAlgoCount};
            cudnnConvolutionFwdAlgoPerf results[] = new cudnnConvolutionFwdAlgoPerf[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            // New way of finding the fastest config
            // Setup for findFastest call
            //             System.out.println("Testing cudnnFindConvolutionForwardAlgorithm ...");
            JCudnn.cudnnFindConvolutionForwardAlgorithm(CudnnHandleManager.getHandle(), xDesc, wDesc, convDesc, dstDesc, requestedAlgoCount, returnedAlgoCountArray, results);
            returnedAlgoCount = returnedAlgoCountArray[0];
            //             for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
            //            	 String result = checkError(results[algoIndex].status);
            //                 System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n",
            //                     results[algoIndex].algo, results[algoIndex].time,
            //                     (long)results[algoIndex].memory, "["+result+"]");
            //             }
            return results[0].algo;
        } else {
            return convAlgorithm;
        }
    }

    public void getWorkSpace() {
        if (this.network.workspace == null) {
            this.network.workspace = new Pointer();
        }
        long most = 0;
        long[] sa = {most};
        handle(JCudnn.cudnnGetConvolutionForwardWorkspaceSize(CudnnHandleManager.getHandle(), yDesc, kernelDesc, convDesc, xDesc, fw_algo, sa));
        if (sa[0] > most) {
            most = sa[0];
        }
        //		System.out.println("bkf_algo:"+bkf_algo);
        handle(JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandleManager.getHandle(), yDesc, xDesc, convDesc, kernelDesc, bkf_algo, sa));
        if (sa[0] > most) {
            most = sa[0];
        }
        //		System.out.println("bkd_algo:"+bkd_algo);
        handle(JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(CudnnHandleManager.getHandle(), kernelDesc, xDesc, convDesc, yDesc, bkd_algo, sa));
        if (sa[0] > most) {
            most = sa[0];
        }
        if (most > this.network.workspaceSize) {
            this.network.workspaceSize = most;
            JCuda.cudaFree(this.network.workspace);
            JCuda.cudaMalloc(this.network.workspace, this.network.workspaceSize);
        }
    }

    @Override
    public void conv(Tensor input, Tensor kernel, Tensor output) {
        // TODO Auto-generated method stub
    }

	@Override
	public void conv(Tensor input, Tensor kernel, Tensor output, int[] pad) {
		// TODO Auto-generated method stub
		
	}
}

