package com.omega.engine.active;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;

/**
 * 
 * @ClassName: Sigmoid
 *
 * @author lijiaming
 *
 * @date 2020年7月24日
 *
 * @Description: sigmoid function
 * 
 * @active: 1 / (1 + exp(-x))
 * 
 * @diff: y(1-y)
 */
public class Sigmoid extends ActiveFunction {
	
	public Sigmoid(){
		this.activeType = ActiveType.sigmoid;
	}
	
	/**
	 * 1 / (1 + exp(-x))
	 * <p>Title: active</p>
	 *
	 * <p>Description: </p>
	 *
	 * @param x
	 * @return
	 *
	 * @see com.omega.engine.active.ActiveFunction#active(double[])
	 */
	@Override
	public double[] active(double[] x) {
		// TODO Auto-generated method stub
		this.input = MatrixUtils.clone(x);
		this.output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			this.output[i] = (double) (1d / (1d + Math.exp(-x[i])));
		}
		return this.output;
	}
	
	/**
	 * y * (1 - y)
	 * <p>Title: diff</p>
	 *
	 * <p>Description: </p>
	 *
	 * @param x
	 * @return
	 *
	 * @see com.omega.engine.active.ActiveFunction#diff(double[])
	 */
	@Override
	public double[] diff() {
		// TODO Auto-generated method stub
		this.diff = MatrixUtils.clone(this.active(this.input));
		for(int i = 0;i<this.output.length;i++) {
			this.diff[i] = (double) this.output[i] * (1d - this.output[i]);
		}
		return this.diff;
	}
	
	@Override
	public double[] activeTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			output[i] = (double) (1d / (1d + Math.exp(-x[i])));
		}
		return output;
	}
	
	@Override
	public double[] diffTemp(double[] x) {
		// TODO Auto-generated method stub
		double[] diff = this.activeTemp(x);
		for(int i = 0;i<x.length;i++) {
			diff[i] = (double) diff[i] * (1d - diff[i]);
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
					this.output2d[c][i][j] = (double) (1d / (1d + Math.exp(-x[c][i][j])));
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
					this.diff2d[c][i][j] = (double) this.output2d[c][i][j] * (1d - this.output2d[c][i][j]);
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
					output2d[c][i][j] = (double) (1d / (1d + Math.exp(-x[c][i][j])));
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
		Sigmoid function = new Sigmoid();
		double[] x = new double[] {-0.033103351676417064,1.0112810799899228,0.0d,0.12};
		function.input = x;
		double error = function.gradientCheck(x);
		System.out.println("error:"+error);
		System.out.println(JsonUtils.toJson(function));
	}

}
