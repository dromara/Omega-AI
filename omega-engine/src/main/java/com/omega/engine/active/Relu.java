package com.omega.engine.active;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;

/**
 * 
 * @ClassName: Relu
 *
 * @author lijiaming
 *
 * @date 2020年7月24日
 *
 * @Description: relu function
 * 
 * @active: y = max(0, x)
 * 
 * @diff: x <0 :dy/dx = 0; 
 *        x >= 0 :dy/dx = 1
 */
public class Relu extends ActiveFunction{
	
	public Relu(){
		this.activeType = ActiveType.relu;
	}
	
	@Override
	public double[] active(double[] x) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(x);
		this.output = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			if(x[i] > 0) {
				this.output[i] = x[i];
			}else {
				this.output[i] = 0;
			}
		}
		return this.output;
	}

	@Override
	public double[] diff() {
		// TODO Auto-generated method stub
		this.diff = MatrixOperation.zero(this.input.length);
		for(int i = 0;i<this.input.length;i++) {
			if(this.input[i] > 0) {
				this.diff[i] = 1;
			}else {
				this.diff[i] = 0;
			}
		}
		return this.diff;
	}
	
	@Override
	public double[] activeTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] output = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			if(x[i] > 0) {
				output[i] = x[i];
			}else {
				output[i] = 0;
			}
		}
		return output;
	}
	
	@Override
	public double[] diffTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] diff = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			if(x[i] > 0) {
				diff[i] = 1;
			}else {
				diff[i] = 0;
			}
		}
		return diff;
	}
	
	@Override
	public double[][][] active(double[][][] x) {
		// TODO Auto-generated method stub
		this.input2d = MatrixOperation.clone(x);
		this.output2d = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					if(x[c][i][j] > 0) {
						this.output2d[c][i][j] = x[c][i][j];
					}else {
						this.output2d[c][i][j] = 0;
					}
				}
			}
		}
		return this.output2d;
	}

	@Override
	public double[][][] diff2d() {
		// TODO Auto-generated method stub
		this.diff2d = MatrixOperation.zero(this.input2d.length,this.input2d[0].length,this.input2d[0][0].length);
		for(int c = 0;c<this.diff2d.length;c++) {
			for(int i = 0;i<this.diff2d[c].length;i++) {
				for(int j = 0;j<this.diff2d[c][i].length;j++) {
					if(this.input2d[c][i][j] > 0) {
						this.diff2d[c][i][j] = 1;
					}else {
						this.diff2d[c][i][j] = 0;
					}
				}
			}
		}
		return this.diff2d;
	}

	@Override
	public double[][][] activeTemp(double[][][] x) {
		// TODO Auto-generated method stub
		double[][][] output2d = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					if(x[c][i][j] > 0) {
						output2d[c][i][j] = x[c][i][j];
					}else {
						output2d[c][i][j] = 0;
					}
				}
			}
		}
		return output2d;
	}

	@Override
	public double[][][] diffTemp(double[][][] x) {
		// TODO Auto-generated method stub
		double[][][] diff2d = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<diff2d.length;c++) {
			for(int i = 0;i<diff2d[c].length;i++) {
				for(int j = 0;j<diff2d[c][i].length;j++) {
					if(x[c][i][j] > 0) {
						diff2d[c][i][j] = 1;
					}else {
						diff2d[c][i][j] = 0;
					}
				}
			}
		}
		return diff2d;
	}
	
	public static void main(String[] args) {
		Relu function = new Relu();
		double[] x = new double[] {0.1,-0.03,0.25,0.4,-0.87,0.12,-0.001,0.0};
		double error = function.gradientCheck(x);
		System.out.println("error:"+error);
		System.out.println(JsonUtils.toJson(function));
	}

}
