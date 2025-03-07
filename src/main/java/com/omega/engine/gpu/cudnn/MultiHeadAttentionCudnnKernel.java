package com.omega.engine.gpu.cudnn;//package com.omega.engine.gpu.cudnn;
//
//import com.omega.common.data.Tensor;
//import com.omega.common.utils.RandomUtils;
//import com.omega.engine.gpu.BaseKernel;
//
//import jcuda.Pointer;
//import jcuda.Sizeof;
//import jcuda.jcudnn.JCudnn;
//import jcuda.jcudnn.cudnnAttnDescriptor;
//import jcuda.jcudnn.cudnnDataType;
//import jcuda.jcudnn.cudnnDropoutDescriptor;
//import jcuda.jcudnn.cudnnMathType;
//import jcuda.jcudnn.cudnnSeqDataDescriptor;
//import jcuda.jcudnn.cudnnTensorDescriptor;
//import jcuda.runtime.JCuda;
//import jcuda.runtime.cudaMemcpyKind;
//import jcuda.jcudnn.cudnnMultiHeadAttnWeightKind;
//import jcuda.jcudnn.cudnnSeqDataAxis;
//
///**
// * MultiHeadAttentionCudnnKernel
// * @author Administrator
// *
// */
//public class MultiHeadAttentionCudnnKernel extends BaseKernel{
//
//	public int attnMode = JCudnn.CUDNN_ATTN_DISABLE_PROJ_BIASES;
//
//	private int headNum = 1;
//
//	private int beamSize = 1;
//
//	private double smScaler = 1.0f;
//
//	public int N;
//
//	public int qSize;  //Q输入维度
//	public int kSize;  //K输入维度
//	public int vSize;  //V输入维度
//
//	public int qProjSize;  //Q每头输出维度 hiddenSize
//	public int kProjSize;  //K每头输出维度 hiddenSize
//	public int vProjSize;  //V每头输出维度 hiddenSize
//	public int oProjSize;  //O多头总输出维度 hiddenSize * headNum
//
//	private int oSize;
//
//	private int seqLenQ;  //最大QO序列数
//	private int seqLenK;  //最大KV序列数
//
//	private boolean resLink = false;
//
//	private Pointer workspace;
//	private Pointer reserveSpace;
//
//	private cudnnAttnDescriptor attnDesc;
//
//	private cudnnTensorDescriptor weightDesc;
//
//	private cudnnDropoutDescriptor dropoutDesc;
//
//	private cudnnSeqDataDescriptor qDesc;
//	private cudnnSeqDataDescriptor kDesc;
//	private cudnnSeqDataDescriptor vDesc;
//	private cudnnSeqDataDescriptor oDesc;
//
//	private int dataType = cudnnDataType.CUDNN_DATA_FLOAT;
//	private int compPrec = cudnnDataType.CUDNN_DATA_FLOAT;
//
//	private int[] qSeqArray;
//	private int[] kSeqArray;
//
//	private int[] loWinIdx;
//	private int[] hiWinIdx;
//
//	private float dropout = 0;
//
//	private long workSize = 0;
//
//	private long reserveSize = 0;
//
//	private long[] weightSpaceSize = { 0 };
//
//	private long sizeWeights;
//
//	public MultiHeadAttentionCudnnKernel(int time,int layerNum,int inputSize,int hiddenSize,float dropout,boolean hasBias) {
//
//		this.dropout = dropout;
//
//		init();
//	}
//
//	public void init() {
//
//		attnDesc = new cudnnAttnDescriptor();
//
//		qDesc = new cudnnSeqDataDescriptor();
//		kDesc = new cudnnSeqDataDescriptor();
//		vDesc = new cudnnSeqDataDescriptor();
//		oDesc = new cudnnSeqDataDescriptor();
//
//        dropoutDesc = new cudnnDropoutDescriptor();
//
//        weightDesc = new cudnnTensorDescriptor();
//
//        workspace = new Pointer();
//        reserveSpace = new Pointer();
//	}
//
//	public void init(int number) {
//
//		if(this.N != number) {
//
//			this.N = number;
//
//			this.smScaler = 1.0f / Math.sqrt(qSize / headNum);
//
//			this.oSize = oProjSize > 0 ? oProjSize : ((vProjSize > 0 ? vProjSize : vSize) * headNum); // ; vProjSize > 0 ? vProjSize * numHeads : vSize;
//
//	        long seed = 1337; // Pick a seed.
//
//	        JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
//
//	        JCudnn.cudnnCreateTensorDescriptor(weightDesc);
//
//	        long stateSizeArray[] = { 0 };
//	        Pointer states = new Pointer();
//	        JCudnn.cudnnDropoutGetStatesSize(CudnnHandleManager.getHandle(), stateSizeArray);
//	        long stateSize = stateSizeArray[0];
//
//	        JCuda.cudaMalloc(states, stateSize);
//
//	        handle(JCudnn.cudnnSetDropoutDescriptor(dropoutDesc,
//	        	CudnnHandleManager.getHandle(),
//	            dropout,
//	            states,
//	            stateSize,
//	            seed));
//
//	        handle(JCudnn.cudnnSetAttnDescriptor(attnDesc,
//	        		attnMode,
//	        		headNum,
//	        		smScaler,
//	        		dataType,
//	        		compPrec,
//	        		cudnnMathType.CUDNN_DEFAULT_MATH,
//	        		null,  //dropoutRate > 0.0 ? drop_desc : NULL,
//	        		null,
//	        		qSize,
//	        		kSize,
//	        		vSize,
//	        		qProjSize,
//	        		kProjSize,
//	        		vProjSize,
//	        		oProjSize,
//	        		seqLenQ,
//	        		seqLenK,
//	        		N,
//	        		beamSize));
//
//	        long[] sizeWeights = {0}, sizeWkspace = {0}, sizeReserve = {0};
//
//	        handle(JCudnn.cudnnGetMultiHeadAttnBuffers(CudnnHandleManager.getHandle(),
//	        		attnDesc,
//	        		sizeWeights,
//	        		sizeWkspace,
//	        		sizeReserve));
//
//	        this.sizeWeights = sizeWeights[0];
//
//	        JCuda.cudaDeviceSynchronize();
//
//	        qSeqArray = new int[N * beamSize];
//	        kSeqArray = new int[N];
//
//	        loWinIdx = new int[seqLenQ];
//	        hiWinIdx = new int[seqLenQ];
//
//	        for (int i = 0; i < N * beamSize; ++i) {
//	            qSeqArray[i] = seqLenQ;
//	        }
//
//	        for (int i = 0; i < N; ++i) {
//	            kSeqArray[i] = seqLenK;
//	        }
//
//	        // Set the maximum attention window in all time-steps.
//	        for (int i = 0; i < seqLenQ; ++i) {
//	            loWinIdx[i] = 0;
//	            hiWinIdx[i] = seqLenQ;
//	        }
//
//	        int dimA[] = new int[4];
//	        int axes[] = new int[4];
//	        axes[3] = cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM;
//	        axes[2] = cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM;
//	        axes[1] = cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM;
//	        axes[0] = cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM;
//
//	        /**
//	         * set query desc
//	         */
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM] = 1;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM] = N;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM] = seqLenQ;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM] = qSize;
//	        JCudnn.cudnnSetSeqDataDescriptor(qDesc, cudnnDataType.CUDNN_DATA_FLOAT, 4, dimA, axes, N, qSeqArray, null);
//
//	        /**
//	         * set key desc
//	         */
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM] = 1;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM] = N;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM] = seqLenK;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM] = kSize;
//	        JCudnn.cudnnSetSeqDataDescriptor(kDesc, cudnnDataType.CUDNN_DATA_FLOAT, 4, dimA, axes, N, kSeqArray, null);
//
//	        /**
//	         * set value desc
//	         */
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM] = 1;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM] = N;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM] = seqLenK;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM] = vSize;
//	        JCudnn.cudnnSetSeqDataDescriptor(vDesc, cudnnDataType.CUDNN_DATA_FLOAT, 4, dimA, axes, N, kSeqArray, null);
//
//	        /**
//	         * set output desc
//	         */
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM] = 1;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM] = N;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM] = seqLenQ;
//	        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM] = oProjSize;
//	        JCudnn.cudnnSetSeqDataDescriptor(oDesc, cudnnDataType.CUDNN_DATA_FLOAT, 4, dimA, axes, N, qSeqArray, null);
//		}
//
//	}
//
//	public void initWeights(Tensor weights) {
//		int[] wKind = {cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_Q_WEIGHTS, cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_K_WEIGHTS, cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_V_WEIGHTS, cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_O_WEIGHTS};
//		for(int i = 0;i<4;i++) {
//			JCudnn.cudnnGetMultiHeadAttnWeights(CudnnHandleManager.getHandle(), attnDesc, wKind[i], sizeWeights, weights.getGpuData(), weightDesc, null);
//		}
//	}
//
//	public void forward(Tensor q,Tensor k,Tensor v,Tensor w,Tensor output) {
//
//		handle(JCudnn.cudnnMultiHeadAttnForward(CudnnHandleManager.getHandle(), attnDesc, -1, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV,
//				qDesc, q.getGpuData(), q.getGpuData(), kDesc, k.getGpuData(), vDesc, v.getGpuData(), oDesc, output.getGpuData(),
//				weightSizeInBytes, w.getGpuData(), workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace));
//
//	}
//
////
////	public void forward(RunModel RUN_MODEL,Tensor input, Tensor hx, Tensor cx, Tensor weight, Tensor output, Tensor hy, Tensor cy) {
////		// TODO Auto-generated method stub
////
////		if(RUN_MODEL == RunModel.TRAIN) {
//////			input.showDM();
////			//cudnnHandle handle, cudnnRNNDescriptor rnnDesc, int fwdMode, Pointer devSeqLengths, cudnnRNNDataDescriptor xDesc, Pointer x,
////			//cudnnRNNDataDescriptor yDesc, Pointer y, cudnnTensorDescriptor hDesc, Pointer hx, Pointer hy, cudnnTensorDescriptor cDesc, Pointer cx,
////			//Pointer cy, long weightSpaceSize, Pointer weightSpace, long workSpaceSize, Pointer workSpace, long reserveSpaceSize, Pointer reserveSpace
////
////			JCudnn.cudnnMultiHeadAttnForward(CudnnHandleManager.getHandle(), attnDesc, -1, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV,
////					qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights,
////					workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
////
////			handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(),
////					rnnDesc,
////					cudnnForwardMode.CUDNN_FWD_MODE_TRAINING,
////					seqP,
////					xDesc,
////					input.getGpuData(),
////					yDesc,
////					output.getGpuData(),
////					hDesc,
////					hx.getGpuData(), //hx
////					hy.getGpuData(), //hy
////					cDesc,
////					cx.getGpuData(), //cx
////					cy.getGpuData(), //cy
////					getWeightSpaceSize()[0],
////					weight.getGpuData(),
////					workSize,
////					workspace,
////					reserveSize,
////					reserveSpace));
////
////		}else {
////
////			handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(),
////					rnnDesc,
////					cudnnForwardMode.CUDNN_FWD_MODE_INFERENCE,
////					seqP,
////					xDesc,
////					input.getGpuData(),
////					yDesc,
////					output.getGpuData(),
////					hDesc,
////					hx.getGpuData(), //hx
////					hy.getGpuData(), //hy
////					cDesc,
////					cx.getGpuData(), //cx
////					cy.getGpuData(), //cy
////					getWeightSpaceSize()[0],
////					weight.getGpuData(),
////					workSize,
////					workspace,
////					reserveSize,
////					reserveSpace));
////		}
////
////	}
////
////	public void dw(Tensor delta, Tensor output, Tensor input, Tensor hx, Tensor dw) {
////		// TODO Auto-generated method stub
////
////		// cudnnRNNBackwardWeights adds to the data in dw.
////		dw.clearGPU();
////
////		JCudnn.cudnnRNNBackwardWeights_v8(CudnnHandleManager.getHandle(),
////				rnnDesc,
////				CUDNN_WGRAD_MODE_ADD,
////				seqP,
////				xDesc,
////				input.getGpuData(),
////				hDesc,
////				hx.getGpuData(), //hx
////				yDesc,
////				output.getGpuData(),
////				dw.getDataLength() * Sizeof.FLOAT,
////				dw.getGpuData(),
////				workSize,
////				workspace,
////				reserveSize,
////				reserveSpace);
////
////	}
////
////	public void dx(Tensor delta,Tensor dhy,Tensor dcy, Tensor output, Tensor hx, Tensor cx, Tensor weight, Tensor diff, Tensor dhx,Tensor dcx) {
////		// TODO Auto-generated method stub
////
////		Pointer dhy_p = null;
////
////		if(dhy != null) {
////			dhy_p = dhy.getGpuData();
////		}
////
////		handle(JCudnn.cudnnRNNBackwardData_v8(CudnnHandleManager.getHandle(),
////				rnnDesc,
////				seqP,
////				yDesc,
////				output.getGpuData(),
////				delta.getGpuData(),
////				xDesc,
////				diff.getGpuData(),
////				hDesc,
////				hx.getGpuData(), //hx
////				dhy_p,  //dhy
////				dhx.getGpuData(),  //dhx
////				cDesc,
////				cx.getGpuData(), //cx
////				dcy.getGpuData(),  //dcy
////				dcx.getGpuData(),  //dcx
////				weight.getDataLength() * Sizeof.FLOAT,
////				weight.getGpuData(),
////				workSize,
////				workspace,
////				reserveSize,
////				reserveSpace));
////
////	}
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
//	public long[] getWeightSpaceSize() {
//		return weightSpaceSize;
//	}
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
//}
