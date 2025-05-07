package com.omega.engine.gpu.cudnn;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAModules;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;

import static jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class TestJCudnn {
    private static cudnnHandle cudnnHandle;

    public static void main(String[] args) {
        //		JCuda.setExceptionsEnabled(true);
        //        JCudnn.setExceptionsEnabled(true);
        //        JCublas2.setExceptionsEnabled(true);
        //        JCudnn.setLogLevel(LogLevel.LOG_DEBUGTRACE);
        CUDAModules.initContext();
        int version = (int) JCudnn.cudnnGetVersion();
        System.out.printf("cudnnGetVersion() : %d , " + "CUDNN_VERSION from cudnn.h : %d\n", version, JCudnn.CUDNN_VERSION);
        GpuHandle(0);
        //	    testBN1D();
        //	    conv();
        conv_transpose();
        //	    System.out.println(new Date().getTime());
    }

    public static void conv_transpose() {
        int convAlgorithm = -1;
        int N = 1;
        int C = 3;
        int H = 1;
        int W = 1;
        int ko = 2;
        int kH = 4;
        int kW = 4;
        int padding = 0;
        int output_padding = 0;
        int dilation = 1;
        int stride = 1;
        int convDims = 2;
        int[] padA = {padding, padding};
        int[] weight = {C, ko, kH, kW};
        int[] upscaleA = {dilation, dilation};
        int[] tensorOuputDimA = {N, C, H, W};
        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor();
        cudnnTensorDescriptor dstDesc = new cudnnTensorDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(xDesc);
        JCudnn.cudnnCreateFilterDescriptor(wDesc);
        JCudnn.cudnnCreateTensorDescriptor(dstDesc);
        JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
        JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, weight);
        int[] filterStrideA = {stride, stride};
        JCudnn.cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        int on = tensorOuputDimA[0];
        int oc = ko;
        //Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        int oh = (H - 1) * stride - 2 * padA[0] + dilation * (kH - 1) + output_padding + 1;
        int ow = (W - 1) * stride - 2 * padA[1] + dilation * (kW - 1) + output_padding + 1;
        System.out.println(on + ":" + oc + ":" + oh + ":" + ow);
        JCudnn.cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
        //		JCudnn.cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
        int fw_algo = getForwardAlgorithm(convAlgorithm, dstDesc, wDesc, convDesc, xDesc);
        int bkf_algo = getBKFGO(convAlgorithm, dstDesc, xDesc, wDesc, convDesc);
        int bkd_algo = getBKDGO(convAlgorithm, dstDesc, xDesc, wDesc, convDesc);
        //		System.out.println(fw_algo);
        /**
         * getWorkspaceSize

         */
        long workspaceSize = 0;
        Pointer workSpace = new Pointer();
        workspaceSize = getConvTransposeWorkSpace(xDesc, wDesc, convDesc, dstDesc, fw_algo, bkf_algo, bkd_algo);
        if (workspaceSize != 0) {
            handle(JCuda.cudaMalloc(workSpace, workspaceSize));
        }
        System.out.println(workspaceSize);
        Pointer alpha_P = Pointer.to(new float[]{1});
        Pointer beta_P = Pointer.to(new float[]{0});
        //	    float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
        float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        //	    float[] grad = MatrixUtils.one(on * oc * oh * ow);
        float[] grad = MatrixUtils.order(on * oc * oh * ow, 1, 1);
        //	    float[] k = RandomUtils.gaussianRandom(ko * C * kH * kW, 0.1f);
        //	    float[] k = MatrixUtils.one(ko * C * kH * kW);
        float[] k = MatrixUtils.order(ko * C * kH * kW, 1, 1);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor kernel = new Tensor(ko, C, kH, kW, k, true);
        Tensor output = new Tensor(on, oc, oh, ow, true);
        Tensor delta = new Tensor(on, oc, oh, ow, grad, true);
        Tensor dw = new Tensor(ko, C, kH, kW, true);
        Tensor dx = new Tensor(N, C, H, W, true);
        //		System.out.println("dw:"+JsonUtils.toJson(dw.syncHost()));
        handle(JCudnn.cudnnConvolutionBackwardData(cudnnHandle, alpha_P, wDesc, kernel.getGpuData(), xDesc, input.getGpuData(), convDesc, bkd_algo, workSpace, workspaceSize, beta_P, dstDesc, output.getGpuData()));
        handle(JCudnn.cudnnConvolutionForward(cudnnHandle, alpha_P, dstDesc, delta.getGpuData(), wDesc, kernel.getGpuData(), convDesc, fw_algo, workSpace, workspaceSize, beta_P, xDesc, dx.getGpuData()));
        handle(JCudnn.cudnnConvolutionBackwardFilter(cudnnHandle, alpha_P, dstDesc, delta.getGpuData(), xDesc, input.getGpuData(), convDesc, bkf_algo, workSpace, workspaceSize, beta_P, wDesc, dw.getGpuData()));
        input.showDM();
        kernel.showDM();
        System.out.println("output:");
        output.showDM();
        System.out.println("delta:");
        delta.showDM();
        System.out.println("dx:");
        dx.showDM();
        System.out.println("dw:");
        dw.showDM();
        if (workspaceSize != 0) {
            JCuda.cudaFree(workSpace);
        }
    }

    public static void conv() {
        int convAlgorithm = -1;
        int N = 128;
        int C = 64;
        int H = 32;
        int W = 32;
        int ko = 128;
        int kH = 3;
        int kW = 3;
        int padding = 2;
        int stride = 1;
        int convDims = 2;
        int[] padA = {padding, padding};
        int[] weight = {ko, C, kH, kW};
        int[] upscaleA = {1, 1};
        int[] tensorOuputDimA = {N, C, H, W};
        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor();
        cudnnTensorDescriptor dstDesc = new cudnnTensorDescriptor();
        //		cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
        /**
         * backward pararms

         */
        cudnnTensorDescriptor dxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor dwDesc = new cudnnFilterDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(xDesc);
        JCudnn.cudnnCreateFilterDescriptor(wDesc);
        JCudnn.cudnnCreateTensorDescriptor(dstDesc);
        JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
        JCudnn.cudnnCreateTensorDescriptor(dxDesc);
        JCudnn.cudnnCreateTensorDescriptor(dyDesc);
        JCudnn.cudnnCreateFilterDescriptor(dwDesc);
        JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, weight);
        JCudnn.cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, weight);
        int[] filterStrideA = {stride, stride};
        JCudnn.cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        handle(JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, xDesc, wDesc, 4, tensorOuputDimA));
        int on = tensorOuputDimA[0];
        int oc = tensorOuputDimA[1];
        int oh = tensorOuputDimA[2];
        int ow = tensorOuputDimA[3];
        System.out.println(on + ":" + oc + ":" + oh + ":" + ow);
        JCudnn.cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
        JCudnn.cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
        int fw_algo = getForwardAlgorithm(convAlgorithm, xDesc, wDesc, convDesc, dstDesc);
        int bkf_algo = getBKFGO(convAlgorithm, xDesc, dyDesc, dwDesc, convDesc);
        int bkd_algo = getBKDGO(convAlgorithm, dxDesc, dyDesc, wDesc, convDesc);
        //		System.out.println(fw_algo);
        /**
         * getWorkspaceSize

         */
        long workspaceSize = 0;
        Pointer workSpace = new Pointer();
        workspaceSize = getWorkSpace(xDesc, dyDesc, wDesc, dwDesc, convDesc, dstDesc, dxDesc, fw_algo, bkf_algo, bkd_algo);
        if (workspaceSize != 0) {
            JCuda.cudaMalloc(workSpace, workspaceSize);
        }
        System.out.println(workspaceSize);
        Pointer alpha_P = Pointer.to(new float[]{1});
        Pointer beta_P = Pointer.to(new float[]{0});
        float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
        float[] grad = RandomUtils.gaussianRandom(on * oc * oh * ow, 0.1f);
        float[] k = RandomUtils.gaussianRandom(ko * C * kH * kW, 0.1f);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor kernel = new Tensor(ko, C, kH, kW, k, true);
        Tensor output = new Tensor(on, oc, oh, ow, true);
        Tensor delta = new Tensor(on, oc, oh, ow, grad, true);
        Tensor dw = new Tensor(ko, C, kH, kW, true);
        Tensor dx = new Tensor(N, C, H, W, true);
        handle(JCudnn.cudnnConvolutionForward(cudnnHandle, alpha_P, xDesc, input.getGpuData(), wDesc, kernel.getGpuData(), convDesc, fw_algo, workSpace, workspaceSize, beta_P, dstDesc, output.getGpuData()));
        //		System.out.println("output:"+JsonUtils.toJson(output.syncHost()));
        handle(JCudnn.cudnnConvolutionBackwardFilter(cudnnHandle, alpha_P, xDesc, input.getGpuData(), dyDesc, delta.getGpuData(), convDesc, bkf_algo, workSpace, workspaceSize, beta_P, dwDesc, dw.getGpuData()));
        //		System.out.println("dw:"+JsonUtils.toJson(dw.syncHost()));
        handle(JCudnn.cudnnConvolutionBackwardData(cudnnHandle, alpha_P, wDesc, kernel.getGpuData(), dyDesc, delta.getGpuData(), convDesc, bkd_algo, workSpace, workspaceSize, beta_P, dxDesc, dx.getGpuData()));
        //		System.out.println("dx:"+JsonUtils.toJson(dx.syncHost()));
        if (workspaceSize != 0) {
            JCuda.cudaFree(workSpace);
        }
    }

    public static long getWorkSpace(cudnnTensorDescriptor xDesc, cudnnTensorDescriptor dyDesc, cudnnFilterDescriptor wDesc, cudnnFilterDescriptor dwDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstDesc, cudnnTensorDescriptor dxDesc, int fw_algo, int bkf_algo, int bkd_algo) {
        long most = 0;
        long[] sa = {most};
        System.out.println("fw_algo:" + fw_algo);
        if (fw_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xDesc, wDesc, convDesc, dstDesc, fw_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        System.out.println("bkf_algo:" + bkf_algo);
        if (bkf_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, xDesc, dyDesc, convDesc, dwDesc, bkf_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        System.out.println("bkd_algo:" + bkd_algo);
        if (bkd_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wDesc, dyDesc, convDesc, dxDesc, bkd_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        return most;
    }

    public static long getConvTransposeWorkSpace(cudnnTensorDescriptor xDesc, cudnnFilterDescriptor wDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstDesc, int fw_algo, int bkf_algo, int bkd_algo) {
        long most = 0;
        long[] sa = {most};
        System.out.println("fw_algo:" + fw_algo);
        if (fw_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, dstDesc, wDesc, convDesc, xDesc, fw_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        System.out.println("bkf_algo:" + bkf_algo);
        if (bkf_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, dstDesc, xDesc, convDesc, wDesc, bkf_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        System.out.println("bkd_algo:" + bkd_algo);
        if (bkd_algo != 9999) {
            handle(JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wDesc, xDesc, convDesc, dstDesc, bkd_algo, sa));
            System.out.println(sa[0]);
            if (sa[0] > most) {
                most = sa[0];
            }
        }
        return most;
    }

    public static int getBKDGO(int convAlgorithm, cudnnTensorDescriptor dxDesc, cudnnTensorDescriptor dyDesc, cudnnFilterDescriptor wDesc, cudnnConvolutionDescriptor convDesc) {
        int requestedAlgoCount = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int returnedAlgoCountArray[] = {returnedAlgoCount};
        cudnnConvolutionBwdDataAlgoPerf results[] = new cudnnConvolutionBwdDataAlgoPerf[2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        System.out.println("Testing cudnnFindConvolutionBackwardDataAlgorithm ...");
        JCudnn.cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCountArray, results);
        returnedAlgoCount = returnedAlgoCountArray[0];
        for (int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
            String result = checkError(results[algoIndex].status);
            System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n", results[algoIndex].algo, results[algoIndex].time, (long) results[algoIndex].memory, "[" + result + "]");
        }
        return results[0].algo;
    }

    public static int getBKFGO(int convAlgorithm, cudnnTensorDescriptor xDesc, cudnnTensorDescriptor dyDesc, cudnnFilterDescriptor dwDesc, cudnnConvolutionDescriptor convDesc) {
        int requestedAlgoCount = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int returnedAlgoCountArray[] = {returnedAlgoCount};
        cudnnConvolutionBwdFilterAlgoPerf results[] = new cudnnConvolutionBwdFilterAlgoPerf[2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        System.out.println("Testing cudnnFindConvolutionBackwardFilterAlgorithm ...");
        JCudnn.cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCountArray, results);
        returnedAlgoCount = returnedAlgoCountArray[0];
        for (int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
            String result = checkError(results[algoIndex].status);
            System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n", results[algoIndex].algo, results[algoIndex].time, (long) results[algoIndex].memory, "[" + result + "]");
        }
        return results[0].algo;
    }

    public static int getForwardAlgorithm(int convAlgorithm, cudnnTensorDescriptor xDesc, cudnnFilterDescriptor wDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstDesc) {
        if (convAlgorithm < 0) {
            int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
            int returnedAlgoCount = -1;
            int returnedAlgoCountArray[] = {returnedAlgoCount};
            cudnnConvolutionFwdAlgoPerf results[] = new cudnnConvolutionFwdAlgoPerf[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            // Choose the best according to the preference
            //             System.out.println("Testing cudnnGetConvolutionForwardAlgorithm_v7 ...");
            //             handle(JCudnn.cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
            //            		 xDesc, wDesc, convDesc,
            //            		 dstDesc, requestedAlgoCount,
            //                 returnedAlgoCountArray,
            //                 results));
            //             returnedAlgoCount = returnedAlgoCountArray[0];
            //             for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex)
            //             {
            //                 handle(results[algoIndex].status);
            //                 System.out.printf("^^^^ for Algo %d: %f time requiring %d memory\n",
            //                      results[algoIndex].algo, results[algoIndex].time,
            //                     (long)results[algoIndex].memory);
            //             }
            // New way of finding the fastest config
            // Setup for findFastest call
            System.out.println("Testing cudnnFindConvolutionForwardAlgorithm ...");
            JCudnn.cudnnFindConvolutionForwardAlgorithm(cudnnHandle, xDesc, wDesc, convDesc, dstDesc, requestedAlgoCount, returnedAlgoCountArray, results);
            returnedAlgoCount = returnedAlgoCountArray[0];
            for (int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
                String result = checkError(results[algoIndex].status);
                System.out.printf("^^^^ for Algo %d: %f time requiring %d memory %s \n", results[algoIndex].algo, results[algoIndex].time, (long) results[algoIndex].memory, "[" + result + "]");
            }
            return results[0].algo;
        } else {
            return convAlgorithm;
        }
    }

    public static void testBN2D() {
        int N = 2;
        int C = 3;
        int H = 5;
        int W = 5;
        double eps = 1e-5;
        double momentum = 0.9f;
        Pointer alpha_P = Pointer.to(new float[]{1});
        Pointer beta_P = Pointer.to(new float[]{0});
        float[] x = new float[]{0.9827f, 0.5268f, 0.4057f, 0.2853f, 0.1708f, 0.4791f, 0.5626f, 0.1290f, 0.9540f, 0.7471f, 0.5806f, 0.8789f, 0.9766f, 0.8142f, 0.9557f, 0.2814f, 0.7667f, 0.5963f, 0.0016f, 0.5944f, 0.4617f, 0.0975f, 0.3558f, 0.3318f, 0.5196f, 0.7558f, 0.7438f, 0.4061f, 0.2737f, 0.1826f, 0.7600f, 0.3608f, 0.3924f, 0.2537f, 0.7536f, 0.7980f, 0.5246f, 0.6428f, 0.0571f, 0.9973f, 0.7106f, 0.5854f, 0.3122f, 0.2741f, 0.2868f, 0.4628f, 0.2696f, 0.0436f, 0.1222f, 0.4933f, 0.5372f, 0.4992f, 0.2837f, 0.8462f, 0.2095f, 0.1916f, 0.1830f, 0.1934f, 0.8305f, 0.0776f, 0.9014f, 0.1835f, 0.7673f, 0.0999f, 0.5783f, 0.7816f, 0.2961f, 0.9230f, 0.3454f, 0.6030f, 0.4821f, 0.0113f, 0.9629f, 0.8698f, 0.8440f, 0.9763f, 0.7661f, 0.2085f, 0.4248f, 0.7407f, 0.5092f, 0.5272f, 0.8521f, 0.1649f, 0.9759f, 0.9084f, 0.3206f, 0.3061f, 0.9648f, 0.3377f, 0.6753f, 0.6662f, 0.4570f, 0.9556f, 0.0918f, 0.8788f, 0.6432f, 0.4928f, 0.8778f, 0.5665f, 0.7979f, 0.5639f, 0.5970f, 0.4987f, 0.1227f, 0.4963f, 0.6865f, 0.5728f, 0.1927f, 0.1199f, 0.5015f, 0.0221f, 0.0826f, 0.0077f, 0.0568f, 0.7569f, 0.7684f, 0.1536f, 0.4406f, 0.2919f, 0.3006f, 0.9501f, 0.1994f, 0.3314f, 0.5612f, 0.3303f, 0.8773f, 0.3262f, 0.1926f, 0.8667f, 0.3360f, 0.5357f, 0.3332f, 0.2044f, 0.5538f, 0.0607f, 0.2203f, 0.7994f, 0.6357f, 0.6469f, 0.8163f, 0.7764f, 0.6821f, 0.6798f, 0.0553f, 0.0609f, 0.2305f, 0.7183f, 0.8135f, 0.7688f};
        float[] gammaData = new float[]{1, 1, 1};
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor gamma = new Tensor(1, C, 1, 1, gammaData, true);
        Tensor beta = new Tensor(1, C, 1, 1, true);
        Tensor mean = new Tensor(1, C, 1, 1, true);
        Tensor var = new Tensor(1, C, 1, 1, true);
        Tensor runingMean = new Tensor(1, C, 1, 1, true);
        Tensor runingVar = new Tensor(1, C, 1, 1, true);
        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor gbmvDesc = new cudnnTensorDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(xDesc);
        JCudnn.cudnnCreateTensorDescriptor(yDesc);
        JCudnn.cudnnCreateTensorDescriptor(gbmvDesc);
        JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetTensor4dDescriptor(gbmvDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);
        handle(JCudnn.cudnnBatchNormalizationForwardTraining(cudnnHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL_PERSISTENT, alpha_P, beta_P, xDesc, input.getGpuData(), yDesc, output.getGpuData(), gbmvDesc, gamma.getGpuData(), beta.getGpuData(), momentum, runingMean.getGpuData(), runingVar.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        System.out.println("mean:" + JsonUtils.toJson(mean.syncHost()));
        System.out.println("var:" + JsonUtils.toJson(var.syncHost()));
        System.out.println("runingMean:" + JsonUtils.toJson(runingMean.syncHost()));
        System.out.println("runingVar:" + JsonUtils.toJson(runingVar.syncHost()));
        System.out.println("output:" + JsonUtils.toJson(output.syncHost()));
        float[] delta_a = MatrixUtils.one(x.length);
        Tensor delta = new Tensor(N, C, H, W, delta_a, true);
        Tensor dx = new Tensor(N, C, H, W, true);
        Tensor dgamma = new Tensor(1, C, 1, 1, true);
        Tensor dbeta = new Tensor(1, C, 1, 1, true);
        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dBnScaleBiasDesc = new cudnnTensorDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(dyDesc);
        JCudnn.cudnnCreateTensorDescriptor(dxDesc);
        JCudnn.cudnnCreateTensorDescriptor(dBnScaleBiasDesc);
        JCudnn.cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
        JCudnn.cudnnSetTensor4dDescriptor(dBnScaleBiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);
        handle(JCudnn.cudnnBatchNormalizationBackward(cudnnHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL_PERSISTENT, alpha_P, beta_P, alpha_P, beta_P, xDesc, input.getGpuData(), dyDesc, delta.getGpuData(), dxDesc, dx.getGpuData(), dBnScaleBiasDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        System.out.println("delta:" + JsonUtils.toJson(delta.syncHost()));
        System.out.println("dgamma:" + JsonUtils.toJson(dgamma.syncHost()));
        System.out.println("dbeta:" + JsonUtils.toJson(dbeta.syncHost()));
        System.out.println("dx:" + JsonUtils.toJson(dx.syncHost()));
        Tensor test_output = new Tensor(N, C, H, W, true);
        handle(JCudnn.cudnnBatchNormalizationForwardInference(cudnnHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL_PERSISTENT, alpha_P, beta_P, xDesc, input.getGpuData(), yDesc, test_output.getGpuData(), gbmvDesc, gamma.getGpuData(), beta.getGpuData(), runingMean.getGpuData(), runingVar.getGpuData(), eps));
        System.out.println("test-output:" + JsonUtils.toJson(test_output.syncHost()));
    }

    public static void testBN1D() {
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 10;
        double eps = 1e-5;
        double momentum = 0.9f;
        Pointer alpha_P = Pointer.to(new float[]{1});
        Pointer beta_P = Pointer.to(new float[]{0});
        float[] x = new float[]{56.773f, -7.231f, 39.634f, 24.728f, -17.959f, 55.251f, -52.316f, -36.322f, -29.619f, 55.24f, 26.773f, -1.231f, 19.634f, 4.728f, 7.958f, -65.251f, 52.316f, -36.322f, -23.619f, -5.247f};
        float[] gammaData = new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor output = new Tensor(N, C, H, W, true);
        Tensor gamma = new Tensor(1, 1, 1, W, gammaData, true);
        Tensor beta = new Tensor(1, 1, 1, W, true);
        Tensor mean = new Tensor(1, 1, 1, W, true);
        Tensor var = new Tensor(1, 1, 1, W, true);
        Tensor runingMean = new Tensor(1, 1, 1, W, true);
        Tensor runingVar = new Tensor(1, 1, 1, W, true);
        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor gbmvDesc = new cudnnTensorDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(xDesc);
        JCudnn.cudnnCreateTensorDescriptor(yDesc);
        JCudnn.cudnnCreateTensorDescriptor(gbmvDesc);
        JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, H, 1);
        JCudnn.cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, H, 1);
        JCudnn.cudnnSetTensor4dDescriptor(gbmvDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, W, 1, 1);
        handle(JCudnn.cudnnBatchNormalizationForwardTraining(cudnnHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL, alpha_P, beta_P, xDesc, input.getGpuData(), yDesc, output.getGpuData(), gbmvDesc, gamma.getGpuData(), beta.getGpuData(), momentum, runingMean.getGpuData(), runingVar.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        System.out.println("mean:" + JsonUtils.toJson(mean.syncHost()));
        System.out.println("var:" + JsonUtils.toJson(var.syncHost()));
        System.out.println("runingMean:" + JsonUtils.toJson(runingMean.syncHost()));
        System.out.println("runingVar:" + JsonUtils.toJson(runingVar.syncHost()));
        System.out.println("output:" + JsonUtils.toJson(output.syncHost()));
        float[] delta_a = MatrixUtils.one(x.length);
        Tensor delta = new Tensor(N, C, H, W, delta_a, true);
        Tensor dx = new Tensor(N, C, H, W, true);
        Tensor dgamma = new Tensor(1, 1, 1, W, true);
        Tensor dbeta = new Tensor(1, 1, 1, W, true);
        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dBnScaleBiasDesc = new cudnnTensorDescriptor();
        JCudnn.cudnnCreateTensorDescriptor(dyDesc);
        JCudnn.cudnnCreateTensorDescriptor(dxDesc);
        JCudnn.cudnnCreateTensorDescriptor(dBnScaleBiasDesc);
        JCudnn.cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, H, 1);
        JCudnn.cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, H, 1);
        JCudnn.cudnnSetTensor4dDescriptor(dBnScaleBiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, W, 1, 1);
        handle(JCudnn.cudnnBatchNormalizationBackward(cudnnHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL, alpha_P, beta_P, alpha_P, alpha_P, xDesc, input.getGpuData(), dyDesc, delta.getGpuData(), dxDesc, dx.getGpuData(), dBnScaleBiasDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
        System.out.println("delta:" + JsonUtils.toJson(delta.syncHost()));
        System.out.println("dgamma:" + JsonUtils.toJson(dgamma.syncHost()));
        System.out.println("dbeta:" + JsonUtils.toJson(dbeta.syncHost()));
        System.out.println("dx:" + JsonUtils.toJson(dx.syncHost()));
    }

    /**
     * Handle.
     *
     * @param returnCode the return run
     */
    public static void handle(final int returnCode) {
        //		  System.out.println(returnCode);
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
        } else {
            System.out.println("success.");
        }
    }

    /**
     * Handle.
     *
     * @param returnCode the return run
     */
    public static String checkError(final int returnCode) {
        //		  System.out.println(returnCode);
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            return jcuda.jcudnn.cudnnStatus.stringFor(returnCode);
        } else {
            return "success";
        }
    }

    /**
     * Instantiates a new Cu dnn.
     *
     * @param deviceNumber the device number
     */
    public static void GpuHandle(int deviceNumber) {
        if (0 <= deviceNumber) {
            initThread();
            cudnnHandle = new cudnnHandle();
            JCudnn.cudnnCreate(cudnnHandle);
        } else {
            cudnnHandle = null;
        }
        //cudaSetDevice();
    }

    /**
     * Init thread.
     */
    public static void initThread() {
        setDevice(0);
    }

    public static void setDevice(final int cudaDeviceId) {
        if (cudaDeviceId < 0)
            throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
        if (!isThreadDeviceId(cudaDeviceId)) {
            final int result = JCuda.cudaSetDevice(cudaDeviceId);
            System.out.println(result);
        }
    }

    public static boolean isThreadDeviceId(int deviceId) {
        Integer integer = getThreadDeviceId();
        return integer != null && (deviceId == integer);
    }

    /**
     * Gets device.
     *
     * @return the device
     */
    public static Integer getThreadDeviceId() {
        return 0;
    }
}

