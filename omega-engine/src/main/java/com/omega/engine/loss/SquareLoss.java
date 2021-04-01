package com.omega.engine.loss;

import com.omega.common.utils.MatrixOperation;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class SquareLoss extends LossFunction {
	
	public final LossType lossType = LossType.square_loss;
	
	private static SquareLoss instance;
	
	public static SquareLoss operation() {
		if(instance == null) {
			instance = new SquareLoss();
		}
		return instance;
	}
	
	@Override
	public double[] loss(double[] x, double[] label) {
		// TODO Auto-generated method stub
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = (x[i] - label[i]) * (x[i] - label[i]) / 2;
		}
		return temp;
	}

	@Override
	public double[] diff(double[] x, double[] label) {
		// TODO Auto-generated method stub
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] - label[i];
		}
		return temp;
	}
	
	public static void main(String[] args) {
		double[] x = new double[] {0.1,-0.03,1.23,-0.4,0.1,-0.12,0.001,0.002};
		double[] label = new double[] {0.1,-0.01,0.022,-0.4,0.803,-0.12,0.001,0.001};
		double error = SquareLoss.operation().gradientCheck(x,label);
		System.out.println("error:"+error);
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.square_loss;
	}
	
}
