package com.omega.engine.nn.network;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.unet.UNetCond;
import com.omega.engine.updater.UpdaterType;

import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Duffsion-UNet
 * @author Administrator
 *
 */
public class DiffusionUNetCond extends Network {

	public int inChannel;
	
	private int[] downChannels;
	
	private int[] midChannels;
	
	private int timeSteps;
	
	private int tEmbDim;
	
	private boolean[] downSamples;
	
	private int numDowns;
	
	private int numMids;
	
	private int numUps;
	
	private boolean[] attns;
	
	private int groups = 32;
	
	private int headNum;
	
	private int convOutChannels;
	
	private int textEmbedDim;
	
	public int maxContextLen;

	private InputLayer inputLayer;
	
	private UNetCond unet;
	
	public int width;
	
	public int height;
	
	public DiffusionUNetCond(LossType lossType,UpdaterType updater,int inChannel,int width,int height,int convOutChannels,int headNum,int[] downChannels,int[] midChannels,boolean[] downSamples,int numDowns,int numMids,int numUps,int timeSteps,int tEmbDim,int textEmbedDim,int maxContextLen,boolean[] attns) {
		this.lossFunction = LossFactory.create(lossType);
		this.updater = updater;
		this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.convOutChannels = convOutChannels;
		this.headNum = headNum;
		this.downChannels = downChannels;
		this.midChannels = midChannels;
		this.numDowns = numDowns;
		this.numMids = numMids;
		this.numUps = numUps;
		this.downSamples = downSamples;
		this.timeSteps = timeSteps;
		this.tEmbDim = tEmbDim;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.attns = attns;
		initLayers();
	}
	
	public void initLayers() {
		
		this.inputLayer = new InputLayer(inChannel, height, width);
		
		unet = new UNetCond(inChannel, inChannel, height, width, downChannels, midChannels, downSamples, attns, timeSteps, tEmbDim, numDowns, numMids, numUps,
				groups, headNum, convOutChannels, textEmbedDim, maxContextLen, this);
		
		this.addLayer(inputLayer);
		this.addLayer(unet);
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		if(layerList.size() <= 1) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		this.setChannel(layerList.get(0).channel);
		this.setHeight(layerList.get(0).height);
		this.setWidth(layerList.get(0).width);
	
		this.oChannel = this.getLastLayer().oChannel;
		this.oHeight = this.getLastLayer().oHeight;
		this.oWidth = this.getLastLayer().oWidth;
		
		if(layerList.get(0).getLayerType() != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}
		
		System.out.println("the network is ready.");
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.DUFFSION_UNET_COND;
	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOutput();
	}
	
	@Override
	public Tensor forward(Tensor input) {
		return null;
	}
	
	public Tensor forward(Tensor input,Tensor t,Tensor context) {
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		this.unet.forward(input, t, context);
		
		return this.unet.getOutput();
	}
	
	public void initBack() {
		
	}
	
	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub
//		lossDiff.showDMByNumber(0);
		
		initBack();
		
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);
		
		this.unet.back(lossDiff);

	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub

		switch (this.getLastLayer().getLayerType()) {
		case softmax:
//			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
//			softmaxLayer.setCurrentLabel(label);
			break;
		case softmax_cross_entropy:
			SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer)this.getLastLayer();
			softmaxWithCrossEntropyLayer.setCurrentLabel(label);
			break;
		default:
			break;
		}
		
		return this.lossFunction.loss(output, label);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		this.clearGrad();
		Tensor t = this.lossFunction.diff(output, label);
//		PrintUtils.printImage(t.data);
		return t;
	}
	
	public void update() {
		
		this.train_time += 1;

		this.unet.update();
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
		/**
		 * forward
		 */
		JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
		
		if(unet.conv_in.cache_delta != null) {
			unet.conv_in.cache_delta.clearGPU();
		}
		
		for(int i = 0;i<unet.downs.size();i++) {
			if(unet.downs.get(i).cache_delta != null) {
				unet.downs.get(i).cache_delta.clearGPU();
			}
		}
		
		JCuda.cudaDeviceSynchronize();
	}

	@Override
	public Tensor loss(Tensor output, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, loss);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		this.clearGrad();
		return this.lossFunction.diff(output, label, diff);
	}
	
	public Tensor loss(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, igonre);
	}

	public Tensor lossDiff(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, igonre);
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
//		unet.saveModel(outputStream);
		System.out.println("tail save success...");
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
//		unet.loadModel(inputStream);
		System.out.println("tail load success...");
	}
	
}
