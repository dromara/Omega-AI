package com.omega.engine.nn.layer;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * SoftmaxWithCrossEntropyLayer
 * 
 * @author Administrator
 * 
 * @remark
 * 
 * The softmax function support only cross entropy loss function now.
 * 
 */
public class SoftmaxWithCrossEntropyLayer extends Layer {
	
	private float[][] currentLabel;
	
	public SoftmaxWithCrossEntropyLayer(int inputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = inputNum;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth,this.output);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, oChannel, oHeight, oWidth, this.diff);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		for(int n = 0;n<this.number;n++) {
			float max = MatrixOperation.max(this.input.maxtir[n][0][0]);
			float[] temp = MatrixOperation.subtraction(this.input.maxtir[n][0][0], max);
			temp = MatrixOperation.exp(temp);
			float sum = MatrixOperation.sum(temp);
			for(int i = 0;i<temp.length;i++) {
				this.output.maxtir[n][0][0][i] = temp[i] / sum;
			}
		}
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		for(int n = 0;n<this.number;n++) {
			for(int w = 0;w<this.oWidth;w++) {
				this.diff.maxtir[n][0][0][w] = this.output.maxtir[n][0][0][w] - this.currentLabel[n][w];
			}
		}
//		MatrixOperation.printImage(this.diff);
//		System.out.println(JsonUtils.toJson(this.diff));
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		
//		System.out.println("====");
//		
//		MatrixOperation.printImage(this.input.maxtir[0]);
		
		this.output();

//		MatrixOperation.printImage(this.output.maxtir[0]);
		
		
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
	}

	public void setCurrentLabel(float[][] currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("softmax with cross layer["+this.index+"]diff start:");
////		
//		MatrixOperation.printImage(this.input);
////		
////		System.out.println("=====================================");
////		
//		MatrixOperation.printImage(this.active);
////		
//		System.out.println("=====================================");
////		
//		MatrixOperation.printImage(this.diff);
////		
//		System.out.println("softmax with cross layer["+this.index+"]diff end.");

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("softmax layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax_cross_entropy;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		
		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		for(int n = 0;n<this.number;n++) {
			float max = MatrixOperation.max(input[n][0][0]);
			float[] temp = MatrixOperation.subtraction(input[n][0][0], max);
			temp = MatrixOperation.exp(temp);
			float sum = MatrixOperation.sum(temp);
			for(int i = 0;i<temp.length;i++) {
				output[n][0][0][i] = temp[i] / sum;
			}
		}
		
		return output;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}

}
