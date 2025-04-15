/*

 * JCuda - Java bindings for NVIDIA CUDA

 *

 * Copyright 2008-2018 Marco Hutter - http://www.jcuda.org

 */
package com.omega.test.rnn.test;

import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.gpu.cudnn.CudnnHandleManager;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaEvent_t;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_BIDIRECTIONAL;
import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_UNIDIRECTIONAL;
import static jcuda.jcudnn.cudnnRNNInputMode.CUDNN_LINEAR_INPUT;
import static jcuda.jcudnn.cudnnRNNMode.*;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

/**
 * A 1:1 port of the RNN example of cuDNN.
 * <p>
 * <p>
 * <p>
 * (Note: This has not been updated for cuDNN 8.0, but even the official
 * <p>
 * example uses several deprecated functions)
 */
public class JCudnnRnnExample {
    public static void main(String[] args) throws Exception {
        CUDAModules.initContext();
        mainImpl(new String[]{"20", "1", "128", "64", "64", "2"});
    }

    public static void mainImpl(String args[]) throws Exception {
        JCuda.setExceptionsEnabled(true);
        JCudnn.setExceptionsEnabled(true);
        int seqLength;
        int numLayers;
        int hiddenSize;
        int inputSize;
        int miniBatch;
        float dropout;
        boolean bidirectional;
        int mode;
        int persistent;
        PrintWriter fp = new PrintWriter(new File("./results.txt"));
        if (args.length == 6) {
            seqLength = Integer.parseInt(args[0]);
            numLayers = Integer.parseInt(args[1]);
            hiddenSize = Integer.parseInt(args[2]);
            inputSize = Integer.parseInt(args[3]);
            miniBatch = Integer.parseInt(args[4]);
            dropout = 0;
            bidirectional = false;
            mode = Integer.parseInt(args[5]);
            persistent = 0;
        } else {
            System.out.printf("Usage:\n");
            System.out.printf("./RNN <seqLength> <numLayers> <hiddenSize> <miniBatch> <mode>\n");
            System.out.printf("Modes: 0 = RNN_RELU, 1 = RNN_TANH, 2 = LSTM, 3 = GRU\n");
            return;
        }
        // -------------------------
        // Create cudnn context
        // -------------------------
        cudnnHandle cudnnHandle = CudnnHandleManager.getHandle();
        //        cudnnCreate(cudnnHandle);
        // -------------------------
        // Set up inputs and outputs
        // -------------------------
        Pointer x = new Pointer();
        Pointer hx = new Pointer();
        Pointer cx = new Pointer();
        Pointer dx = new Pointer();
        Pointer dhx = new Pointer();
        Pointer dcx = new Pointer();
        Pointer y = new Pointer();
        Pointer hy = new Pointer();
        Pointer cy = new Pointer();
        Pointer dy = new Pointer();
        Pointer dhy = new Pointer();
        Pointer dcy = new Pointer();
        // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
        cudaMalloc(x, seqLength * inputSize * miniBatch * Sizeof.FLOAT);
        System.out.println(seqLength * inputSize * miniBatch);
        cudaMalloc(hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(dx, seqLength * inputSize * miniBatch * Sizeof.FLOAT);
        cudaMalloc(dhx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(dcx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(y, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        System.out.println(seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
        cudaMalloc(hy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        System.out.println("hy:" + numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
        cudaMalloc(cy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(dy, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(dhy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        cudaMalloc(dcy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * Sizeof.FLOAT);
        // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
        cudnnTensorDescriptor xDesc[], yDesc[], dxDesc[], dyDesc[];
        cudnnTensorDescriptor hxDesc = new cudnnTensorDescriptor(), cxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor hyDesc = new cudnnTensorDescriptor(), cyDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dhxDesc = new cudnnTensorDescriptor(), dcxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dhyDesc = new cudnnTensorDescriptor(), dcyDesc = new cudnnTensorDescriptor();
        xDesc = new cudnnTensorDescriptor[seqLength];
        yDesc = new cudnnTensorDescriptor[seqLength];
        dxDesc = new cudnnTensorDescriptor[seqLength];
        dyDesc = new cudnnTensorDescriptor[seqLength];
        int dimA[] = new int[3];
        int strideA[] = new int[3];
        // In this example dimA[1] is constant across the whole sequence
        // This isn't required, all that is required is that it does not increase.
        for (int i = 0; i < seqLength; i++) {
            xDesc[i] = new cudnnTensorDescriptor();
            yDesc[i] = new cudnnTensorDescriptor();
            dxDesc[i] = new cudnnTensorDescriptor();
            dyDesc[i] = new cudnnTensorDescriptor();
            cudnnCreateTensorDescriptor(xDesc[i]);
            cudnnCreateTensorDescriptor(yDesc[i]);
            cudnnCreateTensorDescriptor(dxDesc[i]);
            cudnnCreateTensorDescriptor(dyDesc[i]);
            dimA[0] = miniBatch;
            dimA[1] = inputSize;
            dimA[2] = 1;
            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;
            cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
            cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
            dimA[0] = miniBatch;
            dimA[1] = bidirectional ? hiddenSize * 2 : hiddenSize;
            dimA[2] = 1;
            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;
            cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
            cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
        }
        dimA[0] = numLayers * (bidirectional ? 2 : 1);
        dimA[1] = miniBatch;
        dimA[2] = hiddenSize;
        System.out.println(dimA[0] + ":" + dimA[1] + ":" + dimA[2]);
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        cudnnCreateTensorDescriptor(hxDesc);
        cudnnCreateTensorDescriptor(cxDesc);
        cudnnCreateTensorDescriptor(hyDesc);
        cudnnCreateTensorDescriptor(cyDesc);
        cudnnCreateTensorDescriptor(dhxDesc);
        cudnnCreateTensorDescriptor(dcxDesc);
        cudnnCreateTensorDescriptor(dhyDesc);
        cudnnCreateTensorDescriptor(dcyDesc);
        cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
        // -------------------------
        // Set up the dropout descriptor (needed for the RNN descriptor)
        // -------------------------
        long seed = 1337; // Pick a seed.
        cudnnDropoutDescriptor dropoutDesc = new cudnnDropoutDescriptor();
        cudnnCreateDropoutDescriptor(dropoutDesc);
        // How much memory does dropout need for states?
        // These states are used to generate random numbers internally
        // and should not be freed until the RNN descriptor is no longer used
        long stateSizeArray[] = {0};
        Pointer states = new Pointer();
        cudnnDropoutGetStatesSize(cudnnHandle, stateSizeArray);
        long stateSize = stateSizeArray[0];
        cudaMalloc(states, stateSize);
        cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize, seed);
        // -------------------------
        // Set up the RNN descriptor
        // -------------------------
        cudnnRNNDescriptor rnnDesc = new cudnnRNNDescriptor();
        int RNNMode = 0; // cudnnRNNMode
        int RNNAlgo = 0; // cudnnRNNAlgo
        cudnnCreateRNNDescriptor(rnnDesc);
        //        if      (mode == 0) RNNMode = CUDNN_RNN_RELU;
        //        else if (mode == 1) RNNMode = CUDNN_RNN_TANH;
        //        else if (mode == 2) RNNMode = CUDNN_LSTM;
        //        else if (mode == 3) RNNMode = CUDNN_GRU;
        //
        //        // Persistent RNNs are only supported on Pascal+ GPUs.
        //        if      (persistent == 0) RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
        //        else if (persistent == 1) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_STATIC;
        //        else if (persistent == 2) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
        cudnnSetRNNDescriptor_v6(cudnnHandle, rnnDesc, hiddenSize, numLayers, dropoutDesc, CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, RNNMode, RNNAlgo, // Can be changed to use persistent RNNs on Pascal+ GPUs.
                CUDNN_DATA_FLOAT);
        // -------------------------
        // Set up parameters
        // -------------------------
        // This needs to be done after the rnn descriptor is set as otherwise
        // we don't know how many parameters we have to allocate
        Pointer w = new Pointer();
        Pointer dw = new Pointer();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor(), dwDesc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(wDesc);
        cudnnCreateFilterDescriptor(dwDesc);
        long weightsSizeArray[] = {0};
        cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], weightsSizeArray, CUDNN_DATA_FLOAT);
        long weightsSize = weightsSizeArray[0];
        int dimW[] = new int[3];
        dimW[0] = (int) (weightsSize / Sizeof.FLOAT);
        dimW[1] = 1;
        dimW[2] = 1;
        System.out.println("weightsSize:" + weightsSize);
        cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
        cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
        cudaMalloc(w, weightsSize);
        cudaMalloc(dw, weightsSize);
        // -------------------------
        // Set up work space and reserved memory
        // -------------------------
        Pointer workspace = new Pointer();
        Pointer reserveSpace = new Pointer();
        long workSizeArray[] = {0};
        long reserveSizeArray[] = {0};
        // Need for every pass
        cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, workSizeArray);
        long workSize = workSizeArray[0];
        // Only needed in training, shouldn't be touched between passes.
        cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc, reserveSizeArray);
        long reserveSize = reserveSizeArray[0];
        cudaMalloc(workspace, workSize);
        cudaMalloc(reserveSpace, reserveSize);
        // *********************************************************************************************************
        // Initialise weights and inputs
        // *********************************************************************************************************
        // We initialise to something simple.
        // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
        initGPUData(x, seqLength * inputSize * miniBatch, 1.f);
        //        initGPUData(hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        //        initGPUData(cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        initGPUData(dy, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        //        initGPUData(dhy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        //        initGPUData(dcy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        // Weights
        int numLinearLayers = 0;
        if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
            numLinearLayers = 2;
        } else if (RNNMode == CUDNN_LSTM) {
            numLinearLayers = 8;
        } else if (RNNMode == CUDNN_GRU) {
            numLinearLayers = 6;
        }
        for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
            for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
                cudnnFilterDescriptor linLayerMatDesc = new cudnnFilterDescriptor();
                cudnnCreateFilterDescriptor(linLayerMatDesc);
                Pointer linLayerMat = new Pointer();
                cudnnGetRNNLinLayerMatrixParams(cudnnHandle, rnnDesc, layer, xDesc[0], wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);
                int dataTypeArray[] = {0}; // cudnnDataType
                int formatArray[] = {0}; // cudnnTensorFormat
                int nbDimsArray[] = {0};
                int filterDimA[] = new int[3];
                cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, dataTypeArray, formatArray, nbDimsArray, filterDimA);
                initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float) (filterDimA[0] * filterDimA[1] * filterDimA[2]));
                cudnnDestroyFilterDescriptor(linLayerMatDesc);
                cudnnFilterDescriptor linLayerBiasDesc = new cudnnFilterDescriptor();
                cudnnCreateFilterDescriptor(linLayerBiasDesc);
                Pointer linLayerBias = new Pointer();
                cudnnGetRNNLinLayerBiasParams(cudnnHandle, rnnDesc, layer, xDesc[0], wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);
                cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, dataTypeArray, formatArray, nbDimsArray, filterDimA);
                initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);
                cudnnDestroyFilterDescriptor(linLayerBiasDesc);
            }
        }
        // *********************************************************************************************************
        // Dynamic persistent RNN plan (if using this algo)
        // *********************************************************************************************************
        //        cudnnPersistentRNNPlan rnnPlan = new cudnnPersistentRNNPlan();
        //        if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        //            // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
        //            //       minibatch or datatype don't change.
        //            cudnnCreatePersistentRNNPlan(rnnDesc, miniBatch, CUDNN_DATA_FLOAT, rnnPlan);
        //            // Tell calls using this descriptor which plan to use.
        //            cudnnSetPersistentRNNPlan(rnnDesc, rnnPlan);
        //        }
        // *********************************************************************************************************
        // At this point all of the setup is done. We now need to pass through the RNN.
        // *********************************************************************************************************
        cudaDeviceSynchronize();
        cudaEvent_t start = new cudaEvent_t(), stop = new cudaEvent_t();
        float timeForwardArray[] = {0.0f}, timeBackward1Array[] = {0.0f}, timeBackward2Array[] = {0.0f};
        cudaEventCreate(start);
        cudaEventCreate(stop);
        cudaEventRecord(start, null);
        // If we're not training we use this instead
        //        cudnnRNNForwardInference(cudnnHandle,
        //            rnnDesc,
        //            seqLength,
        //            xDesc,
        //            x,
        //            hxDesc,
        //            hx,
        //            cxDesc,
        //            cx,
        //            wDesc,
        //            w,
        //            yDesc,
        //            y,
        //            hyDesc,
        //            hy,
        //            cyDesc,
        //            cy,
        //            workspace,
        //            workSize);
        System.out.println(workSize + ":" + reserveSize + ":" + seqLength);
        System.out.println(rnnDesc);
        handle(JCudnn.cudnnRNNForwardTraining(cudnnHandle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, null, null, wDesc, w, yDesc, y, hyDesc, hy, null, null, workspace, workSize, reserveSpace, reserveSize));
        cudaEventRecord(stop, null);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(timeForwardArray, start, stop);
        float timeForward = timeForwardArray[0];
        cudaEventRecord(start, null);
        cudnnRNNBackwardData(cudnnHandle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSize, reserveSpace, reserveSize);
        cudaEventRecord(stop, null);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(timeBackward1Array, start, stop);
        float timeBackward1 = timeBackward1Array[0];
        cudaEventRecord(start, null);
        // cudnnRNNBackwardWeights adds to the data in dw.
        cudaMemset(dw, 0, weightsSize);
        cudnnRNNBackwardWeights(cudnnHandle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, dyDesc, dy, workspace, workSize, dwDesc, dw, reserveSpace, reserveSize);
        cudaEventRecord(stop, null);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(timeBackward2Array, start, stop);
        float timeBackward2 = timeBackward2Array[0];
        int numMats = 0;
        if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
            numMats = 2;
        } else if (RNNMode == CUDNN_LSTM) {
            numMats = 8;
        } else if (RNNMode == CUDNN_GRU) {
            numMats = 6;
        }
        // Calculate FLOPS
        System.out.printf("Forward: %3.0f GFLOPS\n", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
        System.out.printf("Backward: %3.0f GFLOPS, ", (double) numMats * 4 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
        System.out.printf("(%3.0f GFLOPS), ", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
        System.out.printf("(%3.0f GFLOPS)\n", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));
        // Calculate FLOPS
        fp.printf("Forward: %3.0f GFLOPS\n", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
        fp.printf("Backward: %3.0f GFLOPS, ", (double) numMats * 4 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
        fp.printf("(%3.0f GFLOPS), ", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
        fp.printf("(%3.0f GFLOPS)\n", (double) numMats * 2 * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));
        // Make double-sure everything is finished before we copy for result checking.
        cudaDeviceSynchronize();
        // *********************************************************************************************************
        // Print checksums.
        // *********************************************************************************************************
        if (true) {
            float testOutputi[];
            float testOutputh[];
            float testOutputc[];
            int biDirScale = (bidirectional ? 2 : 1);
            testOutputi = new float[(hiddenSize * seqLength * miniBatch * biDirScale)];
            testOutputh = new float[(hiddenSize * miniBatch * numLayers * biDirScale)];
            testOutputc = new float[(hiddenSize * miniBatch * numLayers * biDirScale)];
            cudaMemcpy(Pointer.to(testOutputi), y, hiddenSize * seqLength * miniBatch * biDirScale * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaMemcpy(Pointer.to(testOutputh), hy, numLayers * hiddenSize * miniBatch * biDirScale * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            if (RNNMode == CUDNN_LSTM)
                cudaMemcpy(Pointer.to(testOutputc), cy, numLayers * hiddenSize * miniBatch * biDirScale * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            double checksumi = 0.f;
            double checksumh = 0.f;
            double checksumc = 0.f;
            for (int m = 0; m < miniBatch; m++) {
                double localSumi = 0;
                double localSumh = 0;
                double localSumc = 0;
                for (int j = 0; j < seqLength; j++) {
                    for (int i = 0; i < hiddenSize * biDirScale; i++) {
                        localSumi += testOutputi[j * miniBatch * hiddenSize * biDirScale + m * hiddenSize * biDirScale + i];
                    }
                }
                for (int j = 0; j < numLayers * biDirScale; j++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        localSumh += testOutputh[j * hiddenSize * miniBatch + m * hiddenSize + i];
                        if (RNNMode == CUDNN_LSTM)
                            localSumc += testOutputc[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
                checksumi += localSumi;
                checksumh += localSumh;
                checksumc += localSumc;
            }
            System.out.printf("i checksum %E     ", checksumi);
            fp.printf("i checksum %E     ", checksumi);
            if (RNNMode == CUDNN_LSTM) {
                System.out.printf("c checksum %E     ", checksumc);
                fp.printf("c checksum %E     ", checksumc);
            }
            System.out.printf("h checksum %E\n", checksumh);
            fp.printf("h checksum %E\n", checksumh);
        }
        if (true) {
            float testOutputdi[];
            float testOutputdh[];
            float testOutputdc[];
            int biDirScale = (bidirectional ? 2 : 1);
            testOutputdi = new float[(inputSize * seqLength * miniBatch)];
            testOutputdh = new float[(hiddenSize * miniBatch * numLayers * biDirScale)];
            testOutputdc = new float[(hiddenSize * miniBatch * numLayers * biDirScale)];
            cudaMemcpy(Pointer.to(testOutputdi), dx, seqLength * miniBatch * inputSize * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaMemcpy(Pointer.to(testOutputdh), dhx, numLayers * hiddenSize * miniBatch * biDirScale * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            if (RNNMode == CUDNN_LSTM)
                cudaMemcpy(Pointer.to(testOutputdc), dcx, numLayers * hiddenSize * miniBatch * biDirScale * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            float checksumdi = 0.f;
            float checksumdh = 0.f;
            float checksumdc = 0.f;
            for (int m = 0; m < miniBatch; m++) {
                double localSumdi = 0;
                double localSumdh = 0;
                double localSumdc = 0;
                for (int j = 0; j < seqLength; j++) {
                    for (int i = 0; i < inputSize; i++) {
                        localSumdi += testOutputdi[j * miniBatch * inputSize + m * inputSize + i];
                    }
                }
                for (int j = 0; j < numLayers * biDirScale; j++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        localSumdh += testOutputdh[j * hiddenSize * miniBatch + m * hiddenSize + i];
                        if (RNNMode == CUDNN_LSTM)
                            localSumdc += testOutputdc[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
                checksumdi += localSumdi;
                checksumdh += localSumdh;
                checksumdc += localSumdc;
            }
            System.out.printf("di checksum %E    ", checksumdi);
            fp.printf("di checksum %E    ", checksumdi);
            if (RNNMode == CUDNN_LSTM) {
                System.out.printf("dc checksum %E    ", checksumdc);
                fp.printf("dc checksum %E    ", checksumdc);
            }
            System.out.printf("dh checksum %E\n", checksumdh);
            fp.printf("dh checksum %E\n", checksumdh);
        }
        if (true) {
            float testOutputdw[];
            testOutputdw = new float[(int) (weightsSize / Sizeof.FLOAT)];
            cudaMemcpy(Pointer.to(testOutputdw), dw, weightsSize, cudaMemcpyDeviceToHost);
            double checksumdw = 0.;
            for (int i = 0; i < weightsSize / Sizeof.FLOAT; i++) {
                checksumdw += testOutputdw[i];
            }
            System.out.printf("dw checksum %E\n", checksumdw);
            fp.printf("dw checksum %E\n", checksumdw);
        }
        fp.flush();
        fp.close();
        //        if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        //            cudnnDestroyPersistentRNNPlan(rnnPlan);
        //        }
        cudaFree(x);
        cudaFree(hx);
        cudaFree(cx);
        cudaFree(y);
        cudaFree(hy);
        cudaFree(cy);
        cudaFree(dx);
        cudaFree(dhx);
        cudaFree(dcx);
        cudaFree(dy);
        cudaFree(dhy);
        cudaFree(dcy);
        cudaFree(workspace);
        cudaFree(reserveSpace);
        cudaFree(w);
        cudaFree(dw);
        //        cudaFree(states);
        for (int i = 0; i < seqLength; i++) {
            cudnnDestroyTensorDescriptor(xDesc[i]);
            cudnnDestroyTensorDescriptor(yDesc[i]);
            cudnnDestroyTensorDescriptor(dxDesc[i]);
            cudnnDestroyTensorDescriptor(dyDesc[i]);
        }
        cudnnDestroyTensorDescriptor(hxDesc);
        cudnnDestroyTensorDescriptor(cxDesc);
        cudnnDestroyTensorDescriptor(hyDesc);
        cudnnDestroyTensorDescriptor(cyDesc);
        cudnnDestroyTensorDescriptor(dhxDesc);
        cudnnDestroyTensorDescriptor(dcxDesc);
        cudnnDestroyTensorDescriptor(dhyDesc);
        cudnnDestroyTensorDescriptor(dcyDesc);
        //        cudnnDestroyDropoutDescriptor(dropoutDesc);
        cudnnDestroyRNNDescriptor(rnnDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyFilterDescriptor(dwDesc);
        cudnnDestroy(cudnnHandle);
    }

    private static void initGPUData(Pointer data, int numElements, float value) {
        // Note: The original sample used a kernel to initialize the memory.
        // Using a host array to fill the memory is less efficient, but does
        // not require any custom kernels, and is done here for brevity.
        float array[] = new float[numElements];
        Arrays.fill(array, value);
        cudaMemcpy(data, Pointer.to(array), numElements * Sizeof.FLOAT, cudaMemcpyHostToDevice);
    }

    public static void handle(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
            throw new RuntimeException(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
        }
    }
}

