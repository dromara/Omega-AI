package com.omega.engine.loss;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;

public abstract class LossFunction {
	
	public LossType lossType;
	
	public double eta = 0.00001d;
	
	public abstract double[] loss(double[] x, double[] label);
	
	public abstract double[] diff(double[] x, double[] label);
	
	public abstract LossType getLossType();
	
	public double gradientCheck(double[] x, double[] label) {
		double[] diff = this.diff(x,label);
		double[] f1 = this.loss(MatrixOperation.add(x, eta),label);
		double[] f2 = this.loss(MatrixOperation.subtraction(x, eta),label);
		double[] temp = MatrixOperation.subtraction(f1, f2);
		temp = MatrixOperation.division(temp, 2 * eta);
		System.out.println("diff:"+JsonUtils.toJson(diff));
		System.out.println("gradientCheck:"+JsonUtils.toJson(temp));
		double[] error = MatrixOperation.subtraction(diff, temp);
		return MatrixOperation.sum(error);
	} 
	
}
