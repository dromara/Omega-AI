package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.Updater;

/**
 * Base Layer
 * @author Administrator
 *
 */
public abstract class Layer {
	
	public Network network;
	
	public int index = 0;
	
	/**
	 * batch number
	 */
	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;
	
	public int width = 0;
	
	public int oChannel = 0;
	
	public int oHeight = 0;
	
	public int oWidth = 0;
	
	public Blob input;
	
	public Blob output;
	
	public Blob diff;
	
	public Blob delta;
	
	public double[][] deltaW;
	
	public double[] deltaB;
	
	public double[][] weight;
	
	public double[] bias;
	
	public boolean hasBias = true;
	
	public double lambda = 0.01d;
	
	public double learnRate = 0.001d;
	
	public double eta = 0.00001d;
	
	public LayerType layerType;
	
	public Updater updater;
	
	public abstract void init();
	
	public abstract void initBack();
	
	public abstract void initParam();

	public abstract void output();
	
//	/**
//	 * use for gradient check
//	 * @param eta
//	 * @return
//	 */
//	public abstract Blob output(double eta);
	
	public abstract Blob getOutput();
	
	public abstract void diff();
	
	public abstract void forward();
	
	public abstract void back();
	
	public abstract void update();
	
	public abstract void showDiff();
	
	public abstract LayerType getLayerType();
	
	public abstract double[][][][] output(double[][][][] input);
	
	public void setUpdater(Updater updater) {
		this.updater = updater;
	}
	
	public void setNetwork(Network network) {
		this.network = network;
	}
	
	public void setIndex(int index) {
		this.index = index;
	}
	
	public LayerInit save() {
		// TODO Auto-generated method stub
		return new LayerInit(this);
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setInput() {

		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.input = Blobs.transform(number, channel, height, width, this.network.getPreLayer(this.index).output);

	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta() {
//		
//		System.out.println(this.getLayerType().toString() + this.index);
//		
//		MatrixOperation.printImage(this.network.getNextLayer(this.index).diff.maxtir[0][0]);
//		
		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.delta = Blobs.transform(number, oChannel, oHeight, oWidth, this.network.getNextLayer(this.index).diff);
		
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta(Blob delta) {
		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.delta = delta;
	}

	/**
	 * 
	 * @Title: gradientCheck
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 * gradientCheck:
	 * (f(x + eta) - f(x - eta)) / (2 * eta) ≈ f'(x)
	 */
	public double gradientCheck() {
		System.out.println("*******************" + this.index + "="+this.getLayerType()+" layer********************");
		
		if(this.index != 0) {

			double[][][][] output1 = this.output(MatrixOperation.add(this.input.maxtir, this.eta));
			
			double[][][][] output2 = this.output(MatrixOperation.subtraction(this.input.maxtir, this.eta));
			
			double[][][][] gradientCheck = MatrixOperation.subtraction(output1, output2);
			
			gradientCheck = MatrixOperation.division(gradientCheck, 2 * this.eta);

			double finalGCError = 0.0d;
			
			if(this.getLayerType()!=LayerType.pooling) {
				
				double[][][][] error = MatrixOperation.subtractionP(this.delta.maxtir, gradientCheck);
//				System.out.println(JsonUtils.toJson(error));
				finalGCError = MatrixOperation.sum(error);
			}
			
			System.out.println("finalGCError:"+finalGCError);
		}
		
		return 0.0d;
	}

}
