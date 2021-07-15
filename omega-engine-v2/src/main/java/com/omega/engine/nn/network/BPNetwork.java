package com.omega.engine.nn.network;

import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.updater.UpdaterType;

/**
 * BackPropagation Neuron NetWok
 * 
 * @author Administrator
 *
 */
public class BPNetwork extends Network{
	
	public BPNetwork(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}
	
	public BPNetwork(LossFunction lossFunction,UpdaterType updater) {
		this.lossFunction = lossFunction;
		this.updater = updater;
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		
		if(layerList.size() <= 0) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		this.channel = layerList.get(0).channel;
		this.height = layerList.get(0).height;
		this.width = layerList.get(0).width;
		
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
//		System.out.println("layers:"+JsonUtils.toJson(this.layerList));
	}
	
	@Override
	public Blob forward(Blob input) {
		// TODO Auto-generated method stub

		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		/**
		 * forward
		 */
		for(int i = 0;i<layerCount;i++) {

			Layer layer = layerList.get(i);
			
			layer.forward();
			
		}
		
		return this.getOuput();
	}
	
	@Override
	public void back(Blob lossDiff) {
		// TODO Auto-generated method stub
		
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);
		
		for(int i = layerCount - 1;i>=0;i--) {

			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;
			
			layer.back();
			
			layer.update();
			
		}
		
	}
	
	@Override
	public Blob loss(Blob output, double[][] label) {
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
	public Blob lossDiff(Blob output, double[][] label) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label);
	}
	
	@Override
	public Blob predict(Blob input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOuput();
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.BP;
	}

}
