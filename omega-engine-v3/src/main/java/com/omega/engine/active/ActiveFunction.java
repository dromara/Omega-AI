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
	
	public float eta = 0.00001f;
	
	public float[] input;
	
	public float[] output;
	
	public float[] diff;
	
	public float[][][] input2d;
	
	public float[][][] output2d;
	
	public float[][][] diff2d;
	
	public ActiveFunction(){}
	
	public abstract float[] active(float[] x);
	
	public abstract float[] diff();
	
	public abstract float[] activeTemp(float[] x);
	
	public abstract float[] diffTemp(float[] x);
	
	public abstract float[][][] active(float[][][] x);
	
	public abstract float[][][][] active(float[][][][] x);
	
	public abstract float[][][] diff2d();
	
	public abstract float[][][][] diff(float[][][][] x);
	
	public abstract float[][][] activeTemp(float[][][] x);
	
	public abstract float[][][] diffTemp(float[][][] x);
	
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
	public float gradientCheck(float[] x) {
		this.input = MatrixUtils.clone(x);
		float[] diff = this.diff();
		float[] f1 = this.active(MatrixOperation.add(x, eta));		
		float[] f2 = this.active(MatrixOperation.subtraction(x, eta));
		float[] temp = MatrixOperation.subtraction(f1, f2);
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
		float[] error = MatrixOperation.subtraction(diff, temp);
		return MatrixOperation.sum(error);
	}
	
}
