package com.omega.engine.nn.network;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.llama.LlamaTransformerDecoder;
import com.omega.engine.updater.UpdaterType;

/**
 * Llama-2
 * @author Administrator
 *
 */
public class Llama3 extends Network {

	public int vocabSize;
	
	public int embedDim;
	
	public int headNum = 8;
	
	public int nKVHeadNum = 8;
	
	public int decoderNum = 1;
	
	private boolean bias = true;
	
	private boolean flashAttention = false;
	
	private boolean dropout;
	
	private InputLayer inputLayer;
	
	private LlamaTransformerDecoder decoder;
	
	private FullyLayer fullyLayer;
	
	public Llama3(LossType lossType,UpdaterType updater,int headNum,int nKVHeadNum,int decoderNum,int vocabSize,int time,int embedDim,boolean bias,boolean dropout) {
		this.lossFunction = LossFactory.create(lossType);
		this.bias = bias;
		this.dropout = dropout;
		this.decoderNum = decoderNum;
		this.updater = updater;
		this.headNum = headNum;
		this.nKVHeadNum = nKVHeadNum;
		this.time = time;
		this.vocabSize = vocabSize;
		this.embedDim = embedDim;
		this.inputLayer = new InputLayer(1, 1, vocabSize);
		this.decoder = new LlamaTransformerDecoder(this.vocabSize, this.decoderNum, this.headNum, this.nKVHeadNum, this.time, this.embedDim, this.bias, this.dropout, this.flashAttention, this);
		this.fullyLayer = new FullyLayer(embedDim, vocabSize, false, this);
		this.addLayer(inputLayer);
		this.addLayer(decoder);
		this.addLayer(fullyLayer);
	}
	
	public Llama3(LossType lossType,UpdaterType updater,int headNum,int nKVHeadNum,int decoderNum,int vocabSize,int time,int embedDim,boolean bias,boolean dropout,boolean flashAttention) {
		this.flashAttention = flashAttention;
		this.lossFunction = LossFactory.create(lossType);
		this.bias = bias;
		this.dropout = dropout;
		this.decoderNum = decoderNum;
		this.updater = updater;
		this.headNum = headNum;
		this.nKVHeadNum = nKVHeadNum;
		this.time = time;
		this.vocabSize = vocabSize;
		this.embedDim = embedDim;
		this.inputLayer = new InputLayer(1, 1, vocabSize);
		this.decoder = new LlamaTransformerDecoder(this.vocabSize, this.decoderNum, this.headNum, this.nKVHeadNum, this.time, this.embedDim, this.bias, this.dropout, this.flashAttention, this);
		this.fullyLayer = new FullyLayer(embedDim, vocabSize, false, this);
		this.addLayer(inputLayer);
		this.addLayer(decoder);
		this.addLayer(fullyLayer);
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		if(layerList.size() <= 0) {
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
		return NetworkType.LLAMA;
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
		// TODO Auto-generated method stub

		return this.getOutput();
	}
	
	public Tensor forward(Tensor cos,Tensor sin,Tensor input) {
//		System.out.println("en_time:"+en_time+",de_time:"+de_time);
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		inputLayer.forward();
//		JCuda.cudaDeviceSynchronize();
//		long start = System.nanoTime();
//		System.err.println("input:");
//		input.showDM();
		decoder.forward(cos, sin, input);
//		System.err.println("decoder:");
//		decoder.getOutput().showDM();
//		JCuda.cudaDeviceSynchronize();
//		System.out.println("forward1:"+(System.nanoTime() - start) / 1e6+"ms.");
//		long start2 = System.nanoTime();
		fullyLayer.forward(decoder.getOutput());
//		System.err.println("fullyLayer:");
//		fullyLayer.getOutput().showDM();
//		JCuda.cudaDeviceSynchronize();
//		System.out.println("forward2:"+(System.nanoTime() - start2) / 1e6+"ms.");
		return this.getOutput();
	}
	
	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub

	}
	
	public void back(Tensor cos,Tensor sin,Tensor lossDiff) {
		// TODO Auto-generated method stub
//		lossDiff.showDMByNumber(0);
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);
		
//		JCuda.cudaDeviceSynchronize();
//		long start2 = System.nanoTime();
		this.fullyLayer.back(lossDiff);
//		System.err.println("---fully:");
//		this.fullyLayer.diff.showDM();
//		JCuda.cudaDeviceSynchronize();
//		System.out.println("backward2:"+(System.nanoTime() - start2) / 1e6+"ms.");
//		JCuda.cudaDeviceSynchronize();
//		long start3 = System.nanoTime();
		this.decoder.back(cos, sin, this.fullyLayer.diff);
//		JCuda.cudaDeviceSynchronize();
//		System.out.println("backward3:"+(System.nanoTime() - start3) / 1e6+"ms.");
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
		Tensor t = this.lossFunction.diff(output, label);
//		PrintUtils.printImage(t.data);
		return t;
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, loss);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
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
		decoder.saveModel(outputStream);
		fullyLayer.saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		decoder.loadModel(inputStream);
		fullyLayer.loadModel(inputStream);
	}
	
}
