package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.updater.Updater;

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

	private double[] currentLabel;
	
	public SoftmaxWithCrossEntropyLayer(int inputNum,Updater updater) {
		this.inputNum = inputNum;
		this.outputNum = inputNum;
		this.updater = updater;
		this.layerType = LayerType.softmax_cross_entropy;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.zero(this.outputNum);
		this.active = MatrixOperation.zero(this.outputNum);
		this.diff = MatrixOperation.zero(this.inputNum);
		this.nextDiff = MatrixOperation.zero(this.outputNum);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
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
		this.output = this.input;
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
		
		this.diff = MatrixOperation.subtraction(this.active, this.currentLabel);
		
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
	}

	public void setCurrentLabel(double[] currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax_cross_entropy;
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
			active[i] = temp[i] / sum; 
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
	
}
