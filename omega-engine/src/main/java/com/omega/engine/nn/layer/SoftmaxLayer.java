package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.updater.Updater;

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
	
	private double[] currentLabel;
	
	public SoftmaxLayer(int inputNum,int outputNum,Updater updater) {
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.updater = updater;
		this.layerType = LayerType.softmax;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.zero(this.outputNum);
		this.active = MatrixOperation.zero(this.outputNum);
		this.diff = MatrixOperation.zero(this.inputNum);
		this.nextDiff = MatrixOperation.zero(this.outputNum);
		this.delta = MatrixOperation.zero(this.outputNum);
		this.deltaW = MatrixOperation.zero(this.inputNum,this.outputNum);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.weight = MatrixOperation.gaussianRandom(this.inputNum, this.outputNum, 0.1);
		this.bias = MatrixOperation.zero(this.outputNum);
	}

	public void input(double[] data) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(data);
	}

	public void nextDiff(double[] data) {
		// TODO Auto-generated method stub
		this.nextDiff = MatrixOperation.clone(data);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		this.output = this.input;
		if(this.input != null) {
			for(int o = 0;o<this.outputNum;o++) {
				for(int i = 0;i<this.inputNum;i++) {
					this.output[o] += this.input[i] * this.weight[i][o];
				}
				this.output[o] += this.bias[o];
			}
		}
	}

	@Override
	public void active() {
		// TODO Auto-generated method stub
		double max = MatrixOperation.max(this.output);
		double[] temp = MatrixOperation.subtraction(this.output, max);
		temp = MatrixOperation.exp(temp);
		double sum = MatrixOperation.sum(temp);
		for(int i = 0;i<temp.length;i++) {
			this.active[i] = temp[i] / sum; 
		}
//		double[] exp = MatrixOperation.exp(this.input);
//		double sum = MatrixOperation.sum(exp);
//		for(int i = 0;i<exp.length;i++) {
//			this.active[i] = exp[i] / sum;
//		}
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

//		double[][] temp = MatrixOperation.zero(this.diff.length, this.diff.length);
//		
//		for(int i = 0;i<this.diff.length;i++) {
//			for(int j = 0;j<this.diff.length;j++) {
//				if(i == j) {
//					temp[i][j] = (1d - this.output[i]) * this.output[i];
//				}else{
//					temp[i][j] = -1d * this.output[i] * this.output[j];
//				}
//				
//				this.diff[i] += this.nextDiff[j] * temp[i][j];
//				
//			}
//		}
		
		this.diff = MatrixOperation.subtraction(this.active, this.currentLabel);
		
//		System.out.println(JsonUtils.toJson(this.diff));
		
		/**
		 * 计算当前层weight梯度
		 * deltaW(i) = delta(i) * input(i)
		 */
		for(int i = 0;i<this.deltaW.length;i++) {
			for(int j = 0;j<this.deltaW[i].length;j++) {
				this.deltaW[i][j] = this.diff[j] * this.input[i];
			}
		}
		
		/**
		 * 计算当前层误差
		 */
		for(int i = 0;i<this.inputNum;i++) {
			for(int j = 0;j<this.outputNum;j++) {
				this.diff[i] += this.diff[j] * this.weight[i][j];
			}
		}
		
//		System.out.println(JsonUtils.toJson(this.diff));
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		this.output();
		this.active();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.diff();
		if(this.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		if(this.updater != null){
			this.updater.update(this);
		}else{
			for(int i = 0;i<this.weight.length;i++) {
				for(int j = 0;j<this.weight[i].length;j++) {
					this.weight[i][j] -= this.learnRate * this.deltaW[i][j];
				}
			}
			for(int i = 0;i<this.outputNum;i++) {
				this.bias[i] -= this.learnRate * this.delta[i];
			}
		}
	}

	public void setCurrentLabel(double[] currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax;
	}

	@Override
	public double[] activeTemp(double[] output) {
		// TODO Auto-generated method stub
		double[] active = new double[output.length];
		double max = MatrixOperation.max(output);
		double[] temp = MatrixOperation.subtraction(output, max);
		temp = MatrixOperation.exp(temp);
		double sum = MatrixOperation.sum(temp);
		for(int i = 0;i<temp.length;i++) {
			this.active[i] = temp[i] / sum; 
		}
		return active;
	}

	@Override
	public double[] diffTemp() {
		// TODO Auto-generated method stub
		return this.diff;
	}

	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		System.out.println("softmax layer["+this.index+"]diff start:");
		
		MatrixOperation.printImage(this.diff);
		
		System.out.println("softmax layer["+this.index+"]diff end.");
	}
	
}
