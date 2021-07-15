package com.omega.engine.active;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;

/**
 * base active function
 * @ClassName: ActiveFunction
 *
 * @author lijiaming
 *
 * @date 2020年7月24日
 *
 * @Description: 
 * TODO(用一句话描述该文件做什么)
 *
 */
public abstract class ActiveFunction {
	
	public ActiveType activeType;
	
	public double eta = 0.00001d;
	
	public double[] input;
	
	public double[] output;
	
	public double[] diff;
	
	public double[][][] input2d;
	
	public double[][][] output2d;
	
	public double[][][] diff2d;
	
	public ActiveFunction(){}
	
	public abstract double[] active(double[] x);
	
	public abstract double[] diff();
	
	public abstract double[] activeTemp(double[] x);
	
	public abstract double[] diffTemp(double[] x);
	
	public abstract double[][][] active(double[][][] x);
	
	public abstract double[][][] diff2d();
	
	public abstract double[][][] activeTemp(double[][][] x);
	
	public abstract double[][][] diffTemp(double[][][] x);
	
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
	public double gradientCheck(double[] x) {
		this.input = MatrixUtils.clone(x);
		double[] diff = this.diff();
		double[] f1 = this.active(MatrixOperation.add(x, eta));		
		double[] f2 = this.active(MatrixOperation.subtraction(x, eta));
		double[] temp = MatrixOperation.subtraction(f1, f2);
		temp = MatrixOperation.division(temp, 2 * eta);
		if(this.activeType == ActiveType.relu) {
			/**
			 * fix zero output diff is zero.
			 */
			for(int i = 0;i<temp.length;i++) {
				if(x[i] == 0) {
					temp[i] = 0;
				}
			}
		}
		System.out.println("diff:"+JsonUtils.toJson(diff));
		System.out.println("gc:"+JsonUtils.toJson(temp));
		double[] error = MatrixOperation.subtraction(diff, temp);
		return MatrixOperation.sum(error);
	}
	
}
