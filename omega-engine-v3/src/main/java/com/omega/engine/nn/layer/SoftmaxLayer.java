package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * SoftmaxLayer
 * 
 * @author Administrator
 * 
 * @remark
 * 
 * The softmax function support only cross entropy loss function now.
 *
 */
public class SoftmaxLayer extends Layer {
	
	public SoftmaxLayer(int inputNum) {
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
				for(int d = 0;d<this.oWidth;d++) {
					
					if(w == d) {
						this.diff.maxtir[n][0][0][w] += this.output.maxtir[n][0][0][w] * (1.0d - this.output.maxtir[n][0][0][w]) * this.delta.maxtir[n][0][0][d]; 
					}else {
						this.diff.maxtir[n][0][0][w] -= this.output.maxtir[n][0][0][w] * this.output.maxtir[n][0][0][d] * this.delta.maxtir[n][0][0][d]; 
					}
					
				}
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

		this.output();

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

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax;
	}

	@Override
	public float gradientCheck() {
		// TODO Auto-generated method stub
		return 0;
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
