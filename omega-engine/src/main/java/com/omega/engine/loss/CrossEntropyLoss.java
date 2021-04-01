package com.omega.engine.loss;

import com.omega.common.utils.MatrixOperation;

/**
 * Cross Entropy loss function
 * 
 * @author Administrator
 *
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 *
 */
public class CrossEntropyLoss extends LossFunction {

	public final LossType lossType = LossType.cross_entropy;
	
	private static CrossEntropyLoss instance;
	
	public static CrossEntropyLoss operation() {
		if(instance == null) {
			instance = new CrossEntropyLoss();
		}
		return instance;
	}
	
	@Override
	public double[] loss(double[] x, double[] label) {
		// TODO Auto-generated method stub
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = - label[i] * Math.log(x[i]);
		}
		return temp;
	}

	@Override
	public double[] diff(double[] x, double[] label) {
		// TODO Auto-generated method stub
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = - label[i] /  x[i];
		}
		return temp;
	}
	
	public double[] label(double[] x, double[] label) {
		// TODO Auto-generated method stub
		return label;
	}
	
	public static void main(String[] args) {
		double[] x = new double[] {0.2,0.7,0.1};
		double[] label = new double[] {0,1,0};
		double error = CrossEntropyLoss.operation().gradientCheck(x,label);
		System.out.println("error:"+error);
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.cross_entropy;
	}

}
