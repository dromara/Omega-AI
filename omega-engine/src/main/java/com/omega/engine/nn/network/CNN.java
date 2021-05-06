package com.omega.engine.nn.network;

import java.util.List;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.SoftmaxLayer;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;

public class CNN extends Network {
	
	public CNN(List<Layer> layers,LossFunction lossFunction) {
		this.layerList = layers;
		this.lossFunction = lossFunction;
		
		if(layerList == null || layerList.size() <= 0) {
			throw new RuntimeException("layer size must greater than 2.");
		}
		
		this.inputNum = layers.get(0).inputNum;
		this.outputNum = layers.get(layers.size() - 1).outputNum;

	}
	
	@Override
	public void init(BaseData baseData) throws Exception {
		// TODO Auto-generated method stub
		if(baseData == null) {
			throw new Exception("trainData is null.");
		}
		
		DataSet trainData = (DataSet) baseData;
		
		if(trainData.dataInput == null || trainData.dataLabel == null) {
			throw new Exception("trainData is null.");
		}
		
		if(layerList.size() <= 0) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		
		if(layerList.get(0).layerType != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).layerType == LayerType.softmax || layerList.get(layerList.size() - 1).layerType == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}
		
		this.setTrainingData(trainData);
		
		/**
		 * 初始化各个层参数
		 */
		int index = 0;
		for(Layer layer:layerList) {
			layer.index = index;
			layer.GRADIENT_CHECK = this.GRADIENT_CHECK;
			layer.init();
			index++;
		}
		
		System.out.println("the network is ready.");
	}

	@Override
	public double[] forward(double[] onceData) {
		// TODO Auto-generated method stub
		
		/**
		 * forward
		 */
		for(int i = 0;i<layerCount;i++) {

			Layer layer = layerList.get(i);
			
			Layer preLayer = null;
			
			switch (layer.getLayerType()) {
			case input:
				
				InputLayer inputLayer = (InputLayer) layer;
				inputLayer.input(onceData);
				inputLayer.forward();
				
				break;
			case conv:

				preLayer = layerList.get(i - 1);
				
				ConvolutionLayer convolutionLayer = (ConvolutionLayer) layer;
				
				double[][][] convInput = null;
				
				if(preLayer.getLayerType() == LayerType.input) {
					InputLayer preInputLayer = (InputLayer) preLayer;
					convInput = MatrixOperation.transform(preLayer.active, preInputLayer.channel, preInputLayer.height, preInputLayer.width);
//					MatrixOperation.printImage(convInput[0]);
				}else if(preLayer.getLayerType() == LayerType.conv) {
					ConvolutionLayer preConvLayer = (ConvolutionLayer) preLayer;
					convInput = preConvLayer.active;
				}else if(preLayer.getLayerType() == LayerType.pooling){
					PoolingLayer prePoolingLayer = (PoolingLayer) preLayer;
					convInput = prePoolingLayer.active;
				}else{
					throw new RuntimeException("Parent layer must be input,conv,pooling for convolution layer.");
				}
				
				convolutionLayer.input(convInput);
				convolutionLayer.forward();

				break;
			case pooling:
				
				preLayer = layerList.get(i - 1);
				
				PoolingLayer poolingLayer = (PoolingLayer) layer;
				
				double[][][] poolingInput = null;
				
				if(preLayer.getLayerType() == LayerType.conv) {
					ConvolutionLayer preConvLayer = (ConvolutionLayer) preLayer;
					poolingInput = preConvLayer.active;
				}else{
					throw new RuntimeException("Parent layer must be conv for pooling layer.");
				}
				
				poolingLayer.input(poolingInput);
				poolingLayer.forward();

				break;
			case full:

				preLayer = layerList.get(i - 1);
				
				FullyLayer fullyLayer = (FullyLayer) layer;
				
				if(preLayer.getLayerType() == LayerType.pooling) {
					PoolingLayer prePoolingLayer = (PoolingLayer) preLayer;
					fullyLayer.input(MatrixOperation.transform(prePoolingLayer.active));
					fullyLayer.forward();
				}else {
					fullyLayer.input(preLayer.active);
					fullyLayer.forward();
				}
				
				break;
			case softmax:

				preLayer = layerList.get(i - 1);
				
				SoftmaxLayer softmaxLayer = (SoftmaxLayer) layer;
				softmaxLayer.input(preLayer.active);
				softmaxLayer.forward();
				
				break;
			case softmax_cross_entropy:

				preLayer = layerList.get(i - 1);
				
				SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) layer;
				softmaxWithCrossEntropyLayer.input(preLayer.active);
				softmaxWithCrossEntropyLayer.forward();
				
				break;
			default:
				
				break;

			}
			
		}
		
		return this.getOuput();
	}

	@Override
	public double[] loss(double[] output, double[] label) {
		// TODO Auto-generated method stub
		
		switch (this.getLastLayer().getLayerType()) {
		case softmax:
			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
			softmaxLayer.setCurrentLabel(label);
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
	public double[] lossDiff(double[] output, double[] label) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label);
	}

	@Override
	public void back(double[] lossDiff) {
		// TODO Auto-generated method stub
		
		for(int i = layerCount - 1;i>=0;i--) {

			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;
			
			Layer preLayer = null;
			
			double[] diff = null;
			
			if(i == layerCount - 1) {
				diff = lossDiff;
			}else {
				preLayer = layerList.get(i + 1);
				diff = preLayer.diff;
			}
			
			switch (layerList.get(i).getLayerType()) {
			case input:
				
				break;
			case conv:
				
				ConvolutionLayer convolutionLayer = (ConvolutionLayer) layer;
				
				if(preLayer.getLayerType() == LayerType.pooling) {
					
					PoolingLayer poolingLayer = (PoolingLayer) preLayer;

					convolutionLayer.nextDiff(poolingLayer.diff);
					
				}else if(preLayer.getLayerType() == LayerType.conv) {
					
					ConvolutionLayer convLayer = (ConvolutionLayer) preLayer;
					
					convolutionLayer.nextDiff(convLayer.diff);
					
				}else {
					throw new RuntimeException("parent layer must be conv layer or fully layer for conv layer.");
				}
				
//				/**
//				 * 全0梯度打印
//				 */
//				if(MatrixOperation.isZero(convolutionLayer.nextDiff)) {
//					for(Layer onceLayer:layerList) {
//						
//						onceLayer.showDiff();
//					}
//				}
//				
				convolutionLayer.back();
				
				convolutionLayer.update();
				
				break;
			case pooling:

				PoolingLayer poolingLayer = (PoolingLayer) layer;
				
				if(preLayer.getLayerType() == LayerType.full) {
					
//					MatrixOperation.printImage(preLayer.diff);
					
					double[][][] fullDiff = MatrixOperation.transform(preLayer.diff, poolingLayer.channel, poolingLayer.oHeight, poolingLayer.oWidth);
					
					poolingLayer.nextDiff(fullDiff);
					
				}else if(preLayer.getLayerType() == LayerType.conv) {
					
					ConvolutionLayer convLayer = (ConvolutionLayer) preLayer;
					
					poolingLayer.nextDiff(convLayer.diff);
					
//					MatrixOperation.printImage(convLayer.diff);
					
				}else {
					throw new RuntimeException("parent layer must be conv layer or fully layer for pooling layer.");
				}
				
				poolingLayer.back();
				
				poolingLayer.update();
				
				break;
			case full:
				
				FullyLayer fullyLayer = (FullyLayer) layer;
				
				fullyLayer.nextDiff(diff);
				
				fullyLayer.back();
				
				fullyLayer.update();
				
				break;
			case softmax:
				
				SoftmaxLayer softmaxLayer = (SoftmaxLayer) layer;
				
				softmaxLayer.nextDiff(diff);
				
				softmaxLayer.back();
				
				softmaxLayer.update();
				
				break;
			case softmax_cross_entropy:
				
				SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) layer;
				
				softmaxWithCrossEntropyLayer.nextDiff(diff);
				
				softmaxWithCrossEntropyLayer.back();
				
				softmaxWithCrossEntropyLayer.update();
				
				break;
				
			default:
				
				break;

			}
			
		}
		
	}

	@Override
	public double test(DataSet testData) {
		// TODO Auto-generated method stub
		double error = 0.0d;
		
		double trueCount = 0;
		
		for(int i = 0;i<testData.dataSize;i++) {
			
			double[] output = this.predict(testData.dataInput[i]);

//			double[] onceError = MatrixOperation.subtraction(output, testData.dataLabel[i]);
			
			String label = testData.labels[i];
			
			String predictLabel = LabelUtils.vectorTolabel(output, testData.labelSet);
			
			if(!label.equals(predictLabel)) {
				System.out.println("index:"+i+"::"+JsonUtils.toJson(output)+"==>predictLabel:"+predictLabel+"==label:"+label+":"+label.equals(predictLabel));
			}else {
				trueCount++;
			}
			
		}
		
		System.out.println("准确率:"+ trueCount / testData.dataSize * 100 +"%");
		return error;
	}

	@Override
	public double[] predict(double[] input) {
		// TODO Auto-generated method stub
		forward(input);
		return this.getOuput();
	}
	
}
