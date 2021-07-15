package com.omega.engine.active;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;

/**
 * 
 * @ClassName: Tanh
 *
 * @author lijiaming
 *
 * @date 2020年7月27日
 *
 * @Description: Tanh function
 * 
 * @active: y = (e^x - e^-x)/(e^x + e^-x)
 * 
 * @diff: 1 - y^2
 */
public class Tanh extends ActiveFunction {
	
	public Tanh(){
		this.activeType = ActiveType.tanh;
	}
	
	@Override
	public double[] active(double[] x) {
		// TODO Auto-generated method stub
		this.input = MatrixUtils.clone(x);
		this.output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			this.output[i] = (Math.exp(x[i]) - Math.exp(-x[i])) / (Math.exp(x[i]) + Math.exp(-x[i]));
		}
		return this.output;
	}

	@Override
	public double[] diff() {
		// TODO Auto-generated method stub
		this.diff = MatrixUtils.clone(this.active(this.input));
		for(int i = 0;i<this.diff.length;i++) {
			this.diff[i] = 1.0d - (this.diff[i] * this.diff[i]);
		}
		return this.diff;
	}
	
	@Override
	public double[] activeTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			output[i] = (Math.exp(x[i]) - Math.exp(-x[i])) / (Math.exp(x[i]) + Math.exp(-x[i]));
		}
		return output;
	}
	
	@Override
	public double[] diffTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] diff = this.activeTemp(x);
		for(int i = 0;i<diff.length;i++) {
			diff[i] = 1.0d - (diff[i] * diff[i]);
		}
		return diff;
	}

	@Override
	public double[][][] active(double[][][] x) {
		// TODO Auto-generated method stub
		this.input2d = MatrixUtils.clone(x);
		this.output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					this.output2d[c][i][j] = (Math.exp(x[c][i][j]) - Math.exp(-x[c][i][j])) / (Math.exp(x[c][i][j]) + Math.exp(-x[c][i][j]));
				}
			}
		}
		return this.output2d;
	}

	@Override
	public double[][][] diff2d() {
		// TODO Auto-generated method stub
		this.diff2d = MatrixUtils.clone(this.active(this.input2d));
		for(int c = 0;c<this.output2d.length;c++) {
			for(int i = 0;i<this.output2d[c].length;i++) {
				for(int j = 0;j<this.output2d[c][i].length;j++) {
					this.diff2d[c][i][j] = 1.0d - (this.output2d[c][i][j] * this.output2d[c][i][j]);
				}
			}
		}
		return this.diff2d;
	}

	@Override
	public double[][][] activeTemp(double[][][] x) {
		// TODO Auto-generated method stub
		double[][][] output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					output2d[c][i][j] = (Math.exp(x[c][i][j]) - Math.exp(-x[c][i][j])) / (Math.exp(x[c][i][j]) + Math.exp(-x[c][i][j]));
				}
			}
		}
		return output2d;
	}

	@Override
	public double[][][] diffTemp(double[][][] x) {
		// TODO Auto-generated method stub
		double[][][] diff2d = MatrixUtils.clone(this.active(x));
		for(int c = 0;c<diff2d.length;c++) {
			for(int i = 0;i<diff2d[c].length;i++) {
				for(int j = 0;j<diff2d[c][i].length;j++) {
					diff2d[c][i][j] = (double) diff2d[c][i][j] * (1d - diff2d[c][i][j]);
				}
			}
		}
		return this.diff2d;
	}

	public static void main(String[] args) {
		Tanh function = new Tanh();
		double[] x = new double[] {0.1,-0.03,0.025,-0.4,0.807,-0.12,0.001,0.002};
		double error = function.gradientCheck(x);
		System.out.println("error:"+error);
		System.out.println(JsonUtils.toJson(function));
	}

}
