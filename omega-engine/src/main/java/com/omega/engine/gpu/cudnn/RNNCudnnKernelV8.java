package com.omega.engine.gpu.cudnn;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.gpu.RNNBaseKernel;
import com.omega.engine.nn.network.RunModel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.*;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnMathType.CUDNN_DEFAULT_MATH;
import static jcuda.jcudnn.cudnnWgradMode.CUDNN_WGRAD_MODE_ADD;

public class RNNCudnnKernelV8 extends RNNBaseKernel {
    public int persistent = 0;
    public int layerNum = 1;
    public int rnnMode;
    public int inputSize;
    public int hiddenSize;
    public int N;
    private boolean bidirectional;
    private Pointer workspace;
    private Pointer reserveSpace;
    private cudnnRNNDescriptor rnnDesc;
    private cudnnRNNDataDescriptor xDesc;
    private cudnnRNNDataDescriptor yDesc;
    private cudnnTensorDescriptor hDesc;
    private cudnnTensorDescriptor cDesc;
    private cudnnDropoutDescriptor dropoutDesc;
    private int CUDARNNAlgo = cudnnRNNAlgo.CUDNN_RNN_ALGO_STANDARD;
    private float dropout = 0;
    private long workSize = 0;
    private long reserveSize = 0;
    private Pointer seqP = null;
    private boolean hasBias = true;
    private long[] weightSpaceSize = {0};
    private int dataType;
    private int mathPrec;
    private int mathType;
    private int bidirectionalScale = 1;
    private int bidmod = cudnnDirectionMode.CUDNN_UNIDIRECTIONAL;
    private int hidTensorSz;

    public RNNCudnnKernelV8(int time, int layerNum, int inputSize, int hiddenSize, boolean bidirectional, int rnnMode, float dropout, boolean hasBias, CUDAManager cudaManager) {
        super(cudaManager);
        this.hasBias = hasBias;
        this.seqLength = time;
        this.dropout = dropout;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.layerNum = layerNum;
        this.bidirectional = bidirectional;
        switch (rnnMode) {
            case 0:
                this.rnnMode = cudnnRNNMode.CUDNN_RNN_RELU;
                break;
            case 1:
                this.rnnMode = cudnnRNNMode.CUDNN_RNN_TANH;
                break;
            case 2:
                this.rnnMode = cudnnRNNMode.CUDNN_LSTM;
                break;
            case 3:
                this.rnnMode = cudnnRNNMode.CUDNN_GRU;
                break;
            default:
                throw new RuntimeException("RNN mode is only support 0:rnn_relu,1:rnn_tanh,1:lstm,2:gru");
        }
        if (this.bidirectional) {
            bidirectionalScale = 2;
            bidmod = cudnnDirectionMode.CUDNN_BIDIRECTIONAL;
        }
        init();
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

    private static void initGPUData(Pointer data, int numElements, float a, float b) {
        // Note: The original sample used a kernel to initialize the memory.
        // Using a host array to fill the memory is less efficient, but does
        // not require any custom kernels, and is done here for brevity.
        float array[] = RandomUtils.order(numElements, a, b);
        JCuda.cudaMemcpy(data, Pointer.to(array), numElements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    public void init() {
        xDesc = new cudnnRNNDataDescriptor();
        yDesc = new cudnnRNNDataDescriptor();
        hDesc = new cudnnTensorDescriptor();
        cDesc = new cudnnTensorDescriptor();
        dropoutDesc = new cudnnDropoutDescriptor();
        rnnDesc = new cudnnRNNDescriptor();
        workspace = new Pointer();
        reserveSpace = new Pointer();
    }

    public void init(int number, int time) {
        if (this.N != number) {
            this.N = number;
            this.seqLength = time;
            int batchSize = this.N / seqLength;
            hidTensorSz = layerNum * batchSize * hiddenSize * bidirectionalScale;
            JCudnn.cudnnCreateRNNDataDescriptor(xDesc);
            JCudnn.cudnnCreateRNNDataDescriptor(yDesc);
            int[] seqLengthArray = new int[batchSize];
            Pointer paddingFill = Pointer.to(new float[]{0.0f});
            MatrixUtils.fill(seqLengthArray, 0, batchSize, seqLength);
            seqP = CUDAMemoryManager.getPointer(batchSize, Sizeof.INT);
            JCuda.cudaMemcpy(seqP, Pointer.to(seqLengthArray), batchSize * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCudnn.cudnnSetRNNDataDescriptor(xDesc, dataType, cudnnRNNDataLayout.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength, batchSize, inputSize, seqLengthArray, paddingFill);
            JCudnn.cudnnSetRNNDataDescriptor(yDesc, dataType, cudnnRNNDataLayout.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength, batchSize, hiddenSize * bidirectionalScale, seqLengthArray, paddingFill);
            JCudnn.cudnnCreateTensorDescriptor(hDesc);
            JCudnn.cudnnCreateTensorDescriptor(cDesc);
            int[] dimHidden = new int[3];
            dimHidden[0] = layerNum * bidirectionalScale;
            dimHidden[1] = batchSize;
            dimHidden[2] = hiddenSize;
            int[] strideHidden = new int[3];
            strideHidden[0] = dimHidden[2] * dimHidden[1];
            strideHidden[1] = dimHidden[2];
            strideHidden[2] = 1;
            JCudnn.cudnnSetTensorNdDescriptor(hDesc, dataType, 3, dimHidden, strideHidden);
            JCudnn.cudnnSetTensorNdDescriptor(cDesc, dataType, 3, dimHidden, strideHidden);
            long seed = 1337; // Pick a seed.
            JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
            long stateSizeArray[] = {0};
            Pointer states = new Pointer();
            JCudnn.cudnnDropoutGetStatesSize(CudnnHandleManager.getHandle(), stateSizeArray);
            long stateSize = stateSizeArray[0];
            JCuda.cudaMalloc(states, stateSize);
            handle(JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, CudnnHandleManager.getHandle(), dropout, states, stateSize, seed));
            // setup RNN descriptor
            JCudnn.cudnnCreateRNNDescriptor(rnnDesc);
            int projSize = hiddenSize; // no recurrent projection
            dataType = cudnnDataType.CUDNN_DATA_FLOAT;
            mathPrec = cudnnDataType.CUDNN_DATA_FLOAT;
            mathType = cudnnMathType.CUDNN_DEFAULT_MATH;
            // Check consistency of parameters
            if ((dataType == cudnnDataType.CUDNN_DATA_HALF && (mathPrec != cudnnDataType.CUDNN_DATA_HALF && mathPrec != CUDNN_DATA_FLOAT)) || (dataType == cudnnDataType.CUDNN_DATA_FLOAT && (mathPrec != cudnnDataType.CUDNN_DATA_FLOAT)) || (dataType == cudnnDataType.CUDNN_DATA_DOUBLE && (mathPrec != cudnnDataType.CUDNN_DATA_DOUBLE))) {
                System.err.println("[ERROR] Inconsistent parameter: dataType does not match mathPrecision!");
            }
            if ((dataType == cudnnDataType.CUDNN_DATA_FLOAT && (mathType != CUDNN_DEFAULT_MATH && mathType != cudnnMathType.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)) || (dataType == cudnnDataType.CUDNN_DATA_DOUBLE && (mathType != CUDNN_DEFAULT_MATH))) {
                System.err.println("[ERROR] Inconsistent parameter: dataType does not match mathType!");
            }
            int biasMode = cudnnRNNBiasMode.CUDNN_RNN_DOUBLE_BIAS;
            if (!hasBias) {
                biasMode = cudnnRNNBiasMode.CUDNN_RNN_NO_BIAS;
            }
            handle(JCudnn.cudnnSetRNNDescriptor_v8(rnnDesc, CUDARNNAlgo, rnnMode, biasMode, bidmod, cudnnRNNInputMode.CUDNN_LINEAR_INPUT, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, layerNum, dropoutDesc, 0));
            // setup work space and reserved memory
            long[] workSpaceSize = {0};
            long[] reserveSpaceSize = {0};
            handle(JCudnn.cudnnGetRNNTempSpaceSizes(CudnnHandleManager.getHandle(), rnnDesc, cudnnForwardMode.CUDNN_FWD_MODE_TRAINING, xDesc, workSpaceSize, reserveSpaceSize));
            workSize = workSpaceSize[0];
            reserveSize = reserveSpaceSize[0];
            JCuda.cudaMalloc(workspace, workSize);
            JCuda.cudaMalloc(reserveSpace, reserveSize);
            //	        System.out.println(workSize+":"+reserveSize);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public long weightSize() {
        JCudnn.cudnnGetRNNWeightSpaceSize(CudnnHandleManager.getHandle(), rnnDesc, getWeightSpaceSize());
        long weightsSize = getWeightSpaceSize()[0];
        //        System.out.println(weightsSize / Sizeof.FLOAT);
        return weightsSize;
    }

    public void initWeights(Tensor w) {
        //		float[] data = new float[] {0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.90000004f,1.0f,1.1f,1.2f,1.3000001f,1.4f,
        //				0.0f,0.2f,0.4f,0.6f,0.8f,1.0f,1.2f,1.4f,1.6f,1.8000001f,2.0f,2.2f,2.4f,2.6000001f,2.8f,3.0f,3.2f,3.4f,3.6000001f,3.8f,4.0f,4.2000003f,4.4f,4.6f,4.8f,
        //				0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.2f,0.2f,0.2f,0.2f};
        //		float[] data = new float[] {0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.90000004f,1.0f,1.1f,1.2f,1.3000001f,1.4f,
        //				0.0f,0.2f,0.4f,0.6f,0.8f,1.0f,1.2f,1.4f,1.6f,1.8000001f,2.0f,2.2f,2.4f,2.6000001f,2.8f,3.0f,3.2f,3.4f,3.6000001f,3.8f,4.0f,4.2000003f,4.4f,4.6f,4.8f};
        //		float[] data = new float[] {0.0f,0.5f,1.0f,0.1f,0.6f,1.1f,0.2f,0.7f,1.2f,0.3f,0.8f,1.3000001f,0.4f,0.90000004f,1.4f,
        //				0.0f,1.0f,2.0f,3.0f,4.0f,0.2f,1.2f,2.2f,3.2f,4.2000003f,0.4f,1.4f,2.4f,3.4f,4.4f,0.6f,1.6f,2.6000001f,3.6000001f,4.6f,0.8f,1.8000001f,2.8f,3.8f,4.8f,
        //				0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.2f,0.2f,0.2f,0.2f};
        //
        //		w.data = data;
        //		w.fill(0.1f);
        //		float[] w1 = new float[] {0.0f,1.0f,2.0f,3.0f,4.0f,0.2f,1.2f,2.2f,3.2f,4.2000003f,0.4f,1.4f,2.4f,3.4f,4.4f,0.6f,1.6f,2.6000001f,3.6000001f,4.6f,0.8f,1.8000001f,2.8f,3.8f,4.8f};
        //		System.out.println(JsonUtils.toJson(MatrixUtils.transpose(w1, 5, 5)));
        //		System.out.println(w.dataLength);
        //
        //		w.hostToDevice();
        //
        //		System.out.println(JsonUtils.toJson(w.syncHost()));
        float stddev = (float) Math.sqrt(2.0d / (inputSize + hiddenSize)); // glorot_uniform like tensorflow
        curandGenerator generator = new curandGenerator();
        JCurand.curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        JCurand.curandSetPseudoRandomGeneratorSeed(generator, 1337);
        JCurand.curandGenerateNormal(generator, w.getGpuData(), w.getDataLength(), 0, stddev);
    }

    public void forward(RunModel RUN_MODEL, Tensor input, Tensor hx, Tensor cx, Tensor weight, Tensor output, Tensor hy, Tensor cy) {
        // TODO Auto-generated method stub
        if (RUN_MODEL == RunModel.TRAIN) {
            //			input.showDM();
            //cudnnHandle handle, cudnnRNNDescriptor rnnDesc, int fwdMode, Pointer devSeqLengths, cudnnRNNDataDescriptor xDesc, Pointer x,
            //cudnnRNNDataDescriptor yDesc, Pointer y, cudnnTensorDescriptor hDesc, Pointer hx, Pointer hy, cudnnTensorDescriptor cDesc, Pointer cx,
            //Pointer cy, long weightSpaceSize, Pointer weightSpace, long workSpaceSize, Pointer workSpace, long reserveSpaceSize, Pointer reserveSpace
            handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(), rnnDesc, cudnnForwardMode.CUDNN_FWD_MODE_TRAINING, seqP, xDesc, input.getGpuData(), yDesc, output.getGpuData(), hDesc, hx.getGpuData(), //hx
                    hy.getGpuData(), //hy
                    cDesc, cx.getGpuData(), //cx
                    cy.getGpuData(), //cy
                    getWeightSpaceSize()[0], weight.getGpuData(), workSize, workspace, reserveSize, reserveSpace));
        } else {
            handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(), rnnDesc, cudnnForwardMode.CUDNN_FWD_MODE_INFERENCE, seqP, xDesc, input.getGpuData(), yDesc, output.getGpuData(), hDesc, hx.getGpuData(), //hx
                    hy.getGpuData(), //hy
                    cDesc, cx.getGpuData(), //cx
                    cy.getGpuData(), //cy
                    getWeightSpaceSize()[0], weight.getGpuData(), workSize, workspace, reserveSize, reserveSpace));
        }
    }

    public void dw(Tensor delta, Tensor output, Tensor input, Tensor hx, Tensor dw) {
        // TODO Auto-generated method stub
        // cudnnRNNBackwardWeights adds to the data in dw.
        dw.clearGPU();
        JCudnn.cudnnRNNBackwardWeights_v8(CudnnHandleManager.getHandle(), rnnDesc, CUDNN_WGRAD_MODE_ADD, seqP, xDesc, input.getGpuData(), hDesc, hx.getGpuData(), //hx
                yDesc, output.getGpuData(), dw.getDataLength() * Sizeof.FLOAT, dw.getGpuData(), workSize, workspace, reserveSize, reserveSpace);
    }

    public void dx(Tensor delta, Tensor dhy, Tensor dcy, Tensor output, Tensor hx, Tensor cx, Tensor weight, Tensor diff, Tensor dhx, Tensor dcx) {
        // TODO Auto-generated method stub
        Pointer dhy_p = null;
        if (dhy != null) {
            dhy_p = dhy.getGpuData();
        }
        handle(JCudnn.cudnnRNNBackwardData_v8(CudnnHandleManager.getHandle(), rnnDesc, seqP, yDesc, output.getGpuData(), delta.getGpuData(), xDesc, diff.getGpuData(), hDesc, hx.getGpuData(), //hx
                dhy_p,  //dhy
                dhx.getGpuData(),  //dhx
                cDesc, cx.getGpuData(), //cx
                dcy.getGpuData(),  //dcy
                dcx.getGpuData(),  //dcx
                weight.getDataLength() * Sizeof.FLOAT, weight.getGpuData(), workSize, workspace, reserveSize, reserveSpace));
    }

    public long[] getWeightSpaceSize() {
        return weightSpaceSize;
    }
}

