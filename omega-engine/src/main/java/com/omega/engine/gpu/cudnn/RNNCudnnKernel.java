//package com.omega.engine.gpu.cudnn;
//
//import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
//import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_BIDIRECTIONAL;
//import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_UNIDIRECTIONAL;
//import static jcuda.jcudnn.cudnnRNNAlgo.CUDNN_RNN_ALGO_STANDARD;
//import static jcuda.jcudnn.cudnnRNNInputMode.CUDNN_LINEAR_INPUT;
//import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
//
//import java.util.Arrays;
//
//import com.omega.common.data.Tensor;
//import com.omega.common.utils.JsonUtils;
//import com.omega.common.utils.RandomUtils;
//import com.omega.engine.nn.layer.gpu.RNNBaseKernel;
//import com.omega.engine.nn.network.RunModel;
//
//import jcuda.Pointer;
//import jcuda.Sizeof;
//import jcuda.jcudnn.JCudnn;
//import jcuda.jcudnn.cudnnDropoutDescriptor;
//import jcuda.jcudnn.cudnnFilterDescriptor;
//import jcuda.jcudnn.cudnnRNNDescriptor;
//import jcuda.jcudnn.cudnnRNNMode;
//import jcuda.jcudnn.cudnnTensorDescriptor;
//import jcuda.jcurand.JCurand;
//import jcuda.jcurand.curandGenerator;
//import jcuda.jcurand.curandRngType;
//import jcuda.runtime.JCuda;
//import jcuda.runtime.cudaMemcpyKind;
//
//
//public class RNNCudnnKernel extends RNNBaseKernel{
//
//	public int persistent = 0;
//
//	public boolean bidirectional = false;
//
//	public int numLinearLayers = 2;
//
//	public int layerNum = 1;
//
//	public int rnnMode;
//
//	public int inputSize;
//
//	public int hiddenSize;
//
//	public int N;
//
//	private int bidLength = 1;
//
//	private Pointer hx;
//	private Pointer cx;
//
//	private Pointer hy;
//	private Pointer cy;
//
//	private Pointer dhx;
//	private Pointer dcx;
//
//	private Pointer dhy;
//	private Pointer dcy;
//
//	private Pointer workspace;
//	private Pointer reserveSpace;
//
//	private cudnnRNNDescriptor rnnDesc;
//
//	private cudnnTensorDescriptor[] xDesc;
//	private cudnnTensorDescriptor[] yDesc;
//	private cudnnTensorDescriptor[] dxDesc;
//	private cudnnTensorDescriptor[] dyDesc;
//
//	private cudnnTensorDescriptor hxDesc;
//	private cudnnTensorDescriptor cxDesc;
//	private cudnnTensorDescriptor hyDesc;
//	private cudnnTensorDescriptor cyDesc;
//
//	private cudnnTensorDescriptor dhxDesc;
//	private cudnnTensorDescriptor dcxDesc;
//	private cudnnTensorDescriptor dhyDesc;
//	private cudnnTensorDescriptor dcyDesc;
//
//	private cudnnDropoutDescriptor dropoutDesc;
//
//	private cudnnFilterDescriptor wDesc;
//	private cudnnFilterDescriptor dwDesc;
//
//	private float dropout = 0;
//
//	private long workSize = 0;
//
//	private long reserveSize = 0;
//
//	public RNNCudnnKernel(int time,int layerNum,int inputSize,int hiddenSize,boolean bidirectional,int rnnMode,float dropout) {
//		this.seqLength = time;
//		this.dropout = dropout;
//		this.inputSize = inputSize;
//		this.hiddenSize = hiddenSize;
//		this.bidirectional = bidirectional;
//		this.layerNum = layerNum;
//		switch (rnnMode) {
//		case 0:
//			this.rnnMode = cudnnRNNMode.CUDNN_RNN_RELU;
//			break;
//		case 1:
//			this.rnnMode = cudnnRNNMode.CUDNN_RNN_TANH;
//			break;
//		case 2:
//			this.rnnMode = cudnnRNNMode.CUDNN_LSTM;
//			break;
//		case 3:
//			this.rnnMode = cudnnRNNMode.CUDNN_GRU;
//			break;
//		default:
//			throw new RuntimeException("RNN mode is only support 0:rnn_relu,1:rnn_tanh,1:lstm,2:gru");
//		}
//		if(bidirectional) {
//			this.bidLength = 2;
//		}
//		System.out.println(this.rnnMode);
//	}
//
//	public void init(int number) {
//
//		if(this.N != number) {
//
//			this.N = number;
//
//			hx = new Pointer();
//			cx = new Pointer();
//
//			dhx = new Pointer();
//			dcx = new Pointer();
//
//			hy = new Pointer();
//			cy = new Pointer();
//
//			dhy = new Pointer();
//			dcy = new Pointer();
//
//			int batchSize = this.N / seqLength;
//
//			JCuda.cudaMalloc(hx, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//			JCuda.cudaMalloc(cx, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//
//			JCuda.cudaMalloc(dhx, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//			JCuda.cudaMalloc(dcx, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//
//			JCuda.cudaMalloc(hy, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//			JCuda.cudaMalloc(cy, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//
//			JCuda.cudaMalloc(dhy, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//			JCuda.cudaMalloc(dcy, layerNum * batchSize * bidLength * hiddenSize * Sizeof.FLOAT);
//
//			xDesc = new cudnnTensorDescriptor[seqLength];
//			yDesc = new cudnnTensorDescriptor[seqLength];
//			dxDesc = new cudnnTensorDescriptor[seqLength];
//			dyDesc = new cudnnTensorDescriptor[seqLength];
//
//			hxDesc = new cudnnTensorDescriptor();
//			cxDesc = new cudnnTensorDescriptor();
//	        hyDesc = new cudnnTensorDescriptor();
//	        cyDesc = new cudnnTensorDescriptor();
//	        dhxDesc = new cudnnTensorDescriptor();
//	        dcxDesc = new cudnnTensorDescriptor();
//	        dhyDesc = new cudnnTensorDescriptor();
//	        dcyDesc = new cudnnTensorDescriptor();
//
//			int dimA[] = new int[3];
//	        int strideA[] = new int[3];
//
//	        for (int i = 0; i < seqLength; i++) {
//
//	            xDesc[i] = new cudnnTensorDescriptor();
//	            yDesc[i] = new cudnnTensorDescriptor();
//	            dxDesc[i] = new cudnnTensorDescriptor();
//	            dyDesc[i] = new cudnnTensorDescriptor();
//
//	            JCudnn.cudnnCreateTensorDescriptor(xDesc[i]);
//	            JCudnn.cudnnCreateTensorDescriptor(yDesc[i]);
//	            JCudnn.cudnnCreateTensorDescriptor(dxDesc[i]);
//	            JCudnn.cudnnCreateTensorDescriptor(dyDesc[i]);
//
//	            dimA[0] = batchSize;
//	            dimA[1] = inputSize;
//	            dimA[2] = 1;
//
//	            strideA[0] = dimA[2] * dimA[1];
//	            strideA[1] = dimA[2];
//	            strideA[2] = 1;
//
//	            JCudnn.cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	            JCudnn.cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
//
//	            dimA[0] = batchSize;
//	            dimA[1] = bidirectional ? hiddenSize * 2 : hiddenSize;
//	            dimA[2] = 1;
//
//	            strideA[0] = dimA[2] * dimA[1];
//	            strideA[1] = dimA[2];
//	            strideA[2] = 1;
//
//	            JCudnn.cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	            JCudnn.cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        }
//
//	        dimA[0] = layerNum * (bidirectional ? 2 : 1);
//	        dimA[1] = batchSize;
//	        dimA[2] = hiddenSize;
//
//	        strideA[0] = dimA[2] * dimA[1];
//	        strideA[1] = dimA[2];
//	        strideA[2] = 1;
//
//	        JCudnn.cudnnCreateTensorDescriptor(hxDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(cxDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(hyDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(cyDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(dhxDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(dcxDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(dhyDesc);
//	        JCudnn.cudnnCreateTensorDescriptor(dcyDesc);
//
//	        JCudnn.cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//	        JCudnn.cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);
//
//	        // -------------------------
//	        // Set up the dropout descriptor (needed for the RNN descriptor)
//	        // -------------------------
//	        long seed = 1337; // Pick a seed.
//
//	        dropoutDesc = new cudnnDropoutDescriptor();
//	        JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
//
//	        long stateSizeArray[] = { 0 };
//	        Pointer states = new Pointer();
//	        JCudnn.cudnnDropoutGetStatesSize(CudnnHandleManager.getHandle(), stateSizeArray);
//	        long stateSize = stateSizeArray[0];
//
//	        JCuda.cudaMalloc(states, stateSize);
//
//	        JCudnn.cudnnSetDropoutDescriptor(dropoutDesc,
//	        	CudnnHandleManager.getHandle(),
//	            dropout,
//	            states,
//	            stateSize,
//	            seed);
//
//	        // -------------------------
//	        // Set up the RNN descriptor
//	        // -------------------------
//	        rnnDesc = new cudnnRNNDescriptor();
//
//	        int RNNAlgo = CUDNN_RNN_ALGO_STANDARD; // cudnnRNNAlgo
//
//	        JCudnn.cudnnCreateRNNDescriptor(rnnDesc);
//
//	        handle(JCudnn.cudnnSetRNNDescriptor_v6(CudnnHandleManager.getHandle(),
//	                rnnDesc,
//	                hiddenSize,
//	                layerNum,
//	                dropoutDesc,
//	                CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
//	                bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
//	                	rnnMode,
//	                	RNNAlgo, // Can be changed to use persistent RNNs on Pascal+ GPUs.
//	                    CUDNN_DATA_FLOAT));
//
//	        // -------------------------
//	        // Set up parameters
//	        // -------------------------
//	        wDesc = new cudnnFilterDescriptor();
//	        dwDesc = new cudnnFilterDescriptor();
//
//	        JCudnn.cudnnCreateFilterDescriptor(wDesc);
//	        JCudnn.cudnnCreateFilterDescriptor(dwDesc);
//
//	        long weightsSize = this.weightSize();
//
//	        int dimW[] = new int[3];
//	        dimW[0] =  (int)(weightsSize / Sizeof.FLOAT);
//	        dimW[1] = 1;
//	        dimW[2] = 1;
//
//	        JCudnn.cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
//	        JCudnn.cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
//
//	        workspace = new Pointer();
//	        reserveSpace = new Pointer();
//
//	        long workSizeArray[] = { 0 };
//	        long reserveSizeArray[] = { 0 };
//
//	        // Need for every pass
//	        JCudnn.cudnnGetRNNWorkspaceSize(CudnnHandleManager.getHandle(), rnnDesc, seqLength, xDesc, workSizeArray);
//	        workSize = workSizeArray[0];
//
//	        // Only needed in training, shouldn't be touched between passes.
//	        JCudnn.cudnnGetRNNTrainingReserveSize(CudnnHandleManager.getHandle(), rnnDesc, seqLength, xDesc, reserveSizeArray);
//	        reserveSize = reserveSizeArray[0];
//
//	        JCuda.cudaMalloc(workspace, workSize);
//	        JCuda.cudaMalloc(reserveSpace, reserveSize);
//
////	        System.out.println(workSize+":"+reserveSize);
//
//	        JCuda.cudaDeviceSynchronize();
//		}
//
//	}
//
//	public long weightSize() {
//		long weightsSizeArray[] = { 0 };
//		JCudnn.cudnnGetRNNParamsSize(CudnnHandleManager.getHandle(), rnnDesc, xDesc[0], weightsSizeArray, CUDNN_DATA_FLOAT);
//        long weightsSize = weightsSizeArray[0];
////        System.out.println(weightsSize);
//        return weightsSize;
//	}
//
//	public void initWeights(Tensor w) {
//
//	   // Weights
//	   if (rnnMode == cudnnRNNMode.CUDNN_RNN_RELU || rnnMode == cudnnRNNMode.CUDNN_RNN_TANH) {
//	      numLinearLayers = 2;
//	   }
//	   else if (rnnMode == cudnnRNNMode.CUDNN_LSTM) {
//	      numLinearLayers = 8;
//	   }
//	   else if (rnnMode == cudnnRNNMode.CUDNN_GRU) {
//	      numLinearLayers = 6;
//	   }
//
//	   for (int layer = 0; layer < layerNum * (bidirectional ? 2 : 1); layer++) {
//           for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
//               cudnnFilterDescriptor linLayerMatDesc = new cudnnFilterDescriptor();
//               JCudnn.cudnnCreateFilterDescriptor(linLayerMatDesc);
//               Pointer linLayerMat = new Pointer();
//
//               handle(JCudnn.cudnnGetRNNLinLayerMatrixParams(CudnnHandleManager.getHandle(),
//                   rnnDesc,
//                   layer,
//                   xDesc[0],
//                   wDesc,
//                   w.getGpuData(),
//                   linLayerID,
//                   linLayerMatDesc,
//                   linLayerMat));
//
//               int dataTypeArray[] = { 0 }; // cudnnDataType
//               int formatArray[] = { 0 }; // cudnnTensorFormat
//               int nbDimsArray[] = { 0 };
//               int filterDimA[] = new int[3];
//
//               handle(JCudnn.cudnnGetFilterNdDescriptor(linLayerMatDesc,
//                   3,
//                   dataTypeArray,
//                   formatArray,
//                   nbDimsArray,
//                   filterDimA));
//
////               initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));
//
//               initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], (linLayerID + 1) * 0.1f, 0);
//
//               JCudnn.cudnnDestroyFilterDescriptor(linLayerMatDesc);
//
//               cudnnFilterDescriptor linLayerBiasDesc = new cudnnFilterDescriptor();
//               JCudnn.cudnnCreateFilterDescriptor(linLayerBiasDesc);
//               Pointer linLayerBias = new Pointer();
//
//               handle(JCudnn.cudnnGetRNNLinLayerBiasParams(CudnnHandleManager.getHandle(),
//                   rnnDesc,
//                   layer,
//                   xDesc[0],
//                   wDesc,
//                   w.getGpuData(),
//                   linLayerID,
//                   linLayerBiasDesc,
//                   linLayerBias));
//
//               handle(JCudnn.cudnnGetFilterNdDescriptor(linLayerBiasDesc,
//                   3,
//                   dataTypeArray,
//                   formatArray,
//                   nbDimsArray,
//                   filterDimA));
//
//               initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], (linLayerID + 1) * 0.1f);
//
//               JCudnn.cudnnDestroyFilterDescriptor(linLayerBiasDesc);
//           }
//       }
//
////		float stddev = (float) Math.sqrt(2.0d / (inputSize + hiddenSize)); // glorot_uniform like tensorflow
////
////		curandGenerator generator = new curandGenerator();
////
////		JCurand.curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
////		JCurand.curandSetPseudoRandomGeneratorSeed(generator, 1L);
////
////		JCurand.curandGenerateNormal(generator, w.getGpuData(), w.getDataLength(), 0, stddev);
//
//		System.out.println( JsonUtils.toJson(w.syncHost()));
//
//	}
//
//	@Override
//	public void forward(RunModel RUN_MODEL,Tensor input, Tensor weight, Tensor output) {
//		// TODO Auto-generated method stub
//
//		handle(JCudnn.cudnnRNNForwardTraining(CudnnHandleManager.getHandle(),
//	            rnnDesc,
//	            seqLength,
//	            xDesc,
//	            input.getGpuData(),
//	            hxDesc,
//	            hx,
//	            cxDesc,
//	            cx,
//	            wDesc,
//	            weight.getGpuData(),
//	            yDesc,
//	            output.getGpuData(),
//	            hyDesc,
//	            hy,
//	            cyDesc,
//	            cy,
//	            workspace,
//	            workSize,
//	            reserveSpace,
//	            reserveSize));
//
//	}
//
//	@Override
//	public void dw(Tensor delta, Tensor output, Tensor input, Tensor dw) {
//		// TODO Auto-generated method stub
//
//		// cudnnRNNBackwardWeights adds to the data in dw.
//		dw.clearGPU();
//
////		JCudnn.cudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, dhx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
//
//		handle(JCudnn.cudnnRNNBackwardWeights(CudnnHandleManager.getHandle(),
//	            rnnDesc,
//	            seqLength,
//	            xDesc,
//	            input.getGpuData(),
//	            hxDesc,
//	            hx,
//	            yDesc,
//	            output.getGpuData(),
//	            workspace,
//	            workSize,
//	            dwDesc,
//	            dw.getGpuData(),
//	            reserveSpace,
//	            reserveSize));
//
//	}
//
//	@Override
//	public void dx(Tensor delta, Tensor output, Tensor weight, Tensor diff) {
//		// TODO Auto-generated method stub
//		handle(JCudnn.cudnnRNNBackwardData(CudnnHandleManager.getHandle(),
//	            rnnDesc,
//	            seqLength,
//	            yDesc,
//	            output.getGpuData(),
//	            dyDesc,
//	            delta.getGpuData(),
//	            dhyDesc,
//	            dhy,
//	            dcyDesc,
//	            dcy,
//	            wDesc,
//	            weight.getGpuData(),
//	            hxDesc,
//	            hx,
//	            cxDesc,
//	            cx,
//	            dxDesc,
//	            diff.getGpuData(),
//	            dhxDesc,
//	            dhx,
//	            dcxDesc,
//	            dcx,
//	            workspace,
//	            workSize,
//	            reserveSpace,
//	            reserveSize));
//
//	}
//
//	/**
//	 * Handle.
//	 *
//	 * @param returnCode the return run
//	 */
//	public static void handle(final int returnCode) {
//		if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
//		      System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
//		      throw new RuntimeException(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
//		}
//	}
//
//	public static String checkError(final int returnCode) {
//	    if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
//	        return jcuda.jcudnn.cudnnStatus.stringFor(returnCode);
//	    }else {
//	    	return "success";
//	    }
//	}
//
//	private static void initGPUData(Pointer data, int numElements, float value){
//        // Note: The original sample used a kernel to initialize the memory.
//        // Using a host array to fill the memory is less efficient, but does
//        // not require any custom kernels, and is done here for brevity.
//        float array[] = new float[numElements];
//        Arrays.fill(array, value);
//        JCuda.cudaMemcpy(data, Pointer.to(array),
//            numElements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
//    }
//
//	private static void initGPUData(Pointer data, int numElements, float a, float b){
//        // Note: The original sample used a kernel to initialize the memory.
//        // Using a host array to fill the memory is less efficient, but does
//        // not require any custom kernels, and is done here for brevity.
//        float array[] = RandomUtils.order(numElements, a, b);
//        JCuda.cudaMemcpy(data, Pointer.to(array),
//            numElements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
//    }
//
//	@Override
//	public void forward(RunModel RUN_MODEL, Tensor input, Tensor hidden, Tensor weight, Tensor output) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void forward(RunModel RUN_MODEL, Tensor input, Tensor hiddenX, Tensor hiddenY, Tensor weight,
//			Tensor output) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void dw(Tensor delta, Tensor output, Tensor input, Tensor hiddenX, Tensor dw) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void dx(Tensor delta, Tensor output, Tensor hiddenX, Tensor weight, Tensor diff) {
//		// TODO Auto-generated method stub
//
//	}
//
//}

