package com.omega.engine.nn.layer;

import java.util.List;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.data.CacheDataSet;
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
	
	public Layer parent;
	
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
	
	public float[][] deltaW;
	
	public float[] deltaB;
	
	public float[][] weight;
	
	public float[] bias;
	
	public boolean hasBias = true;
	
	public float lambda = 0.01f;
	
	public float learnRate = 0.001f;
	
	public float eta = 0.00001f;
	
	public LayerType layerType;
	
	public Updater updater;
	
	public boolean isIdentity = false;
	
	/**
	 * cache data
	 */
	private CacheDataSet tampDataSet;
	

	public List<Layer> layers = null;
	
	
	public abstract void init();
	
	public abstract void initBack();
	
	public abstract void initParam();

	public abstract void output();
	
//	/**
//	 * use for gradient check
//	 * @param eta
//	 * @return
//	 */
//	public abstract Blob output(float eta);
	
	public abstract Blob getOutput();
	
	public abstract void diff();
	
	public abstract void forward();
	
	public abstract void back();
	
	public abstract void update();
	
	public abstract void showDiff();
	
	public abstract LayerType getLayerType();
	
	public abstract float[][][][] output(float[][][][] input);
	
	public abstract void initCache();
	
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
		
		if(parent == null) {

			/**
			 * 获取上一层的输出作为当前层的输入
			 */
			this.input = Blobs.transform(number, channel, height, width, this.network.getPreLayer(this.index).output);

		}else {
			/**
			 * resnet block layer
			 */
			if(this.index == 0) {
				this.input = parent.input;
			}else {
				this.input = this.parent.layers.get(index - 1).output;
			}
			
		}
		
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

		if(parent == null) {

			/**
			 * 获取上一层的输出作为当前层的输入
			 */
			this.delta = Blobs.transform(number, oChannel, oHeight, oWidth, this.network.getNextLayer(this.index).diff);

		}else {
			/**
			 * resnet block layer
			 */
			if(this.index == parent.layers.size() - 1 || isIdentity) {
				this.delta = parent.delta;
			}else {
				this.delta = Blobs.transform(number, oChannel, oHeight, oWidth, parent.layers.get(index + 1).diff);
			}
			
		}
		
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
	public float gradientCheck() {
		System.out.println("*******************" + this.index + "="+this.getLayerType()+" layer********************");
		
		if(this.index != 0) {

			float[][][][] output1 = this.output(MatrixOperation.add(this.input.maxtir, this.eta));
			
			float[][][][] output2 = this.output(MatrixOperation.subtraction(this.input.maxtir, this.eta));
			
			float[][][][] gradientCheck = MatrixOperation.subtraction(output1, output2);
			
			gradientCheck = MatrixOperation.division(gradientCheck, 2 * this.eta);

			float finalGCError = 0.0f;
			
			if(this.getLayerType()!=LayerType.pooling) {
				
				float[][][][] error = MatrixOperation.subtractionP(this.delta.maxtir, gradientCheck);
//				System.out.println(JsonUtils.toJson(error));
				finalGCError = MatrixOperation.sum(error);
			}
			
			System.out.println("finalGCError:"+finalGCError);
		}
		
		return 0.0f;
	}

	public CacheDataSet getTampDataSet() {
		return tampDataSet;
	}

	public void setTampDataSet(CacheDataSet tampDataSet) {
		this.tampDataSet = tampDataSet;
	}

}
