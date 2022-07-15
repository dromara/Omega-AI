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
	 * @see com.omega.engine.active.ActiveFunction#active(float[])
	 */
	@Override
	public float[] active(float[] x) {
		// TODO Auto-generated method stub
		this.input = MatrixUtils.clone(x);
		this.output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			this.output[i] = 1.0f / (1.0f + (float)Math.exp(-x[i]));
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
	 * @see com.omega.engine.active.ActiveFunction#diff(float[])
	 */
	@Override
	public float[] diff() {
		// TODO Auto-generated method stub
		this.diff = MatrixUtils.clone(this.active(this.input));
		for(int i = 0;i<this.output.length;i++) {
			this.diff[i] = this.output[i] * (1.0f - this.output[i]);
		}
		return this.diff;
	}
	
	@Override
	public float[] activeTemp(float[] x) {
		// TODO Auto-generated method stub
		float[] output = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			output[i] = 1f / (1f + (float)Math.exp(-x[i]));
		}
		return output;
	}
	
	@Override
	public float[] diffTemp(float[] x) {
		// TODO Auto-generated method stub
		float[] diff = this.activeTemp(x);
		for(int i = 0;i<x.length;i++) {
			diff[i] = diff[i] * (1.0f - diff[i]);
		}
		return diff;
	}

	@Override
	public float[][][] active(float[][][] x) {
		// TODO Auto-generated method stub
		this.input2d = MatrixUtils.clone(x);
		this.output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					this.output2d[c][i][j] = 1.0f / (1.0f +  (float)Math.exp(-x[c][i][j]));
				}
			}
		}
		return this.output2d;
	}

	@Override
	public float[][][][] active(float[][][][] x) {
		// TODO Auto-generated method stub
		float[][][][] r = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						r[n][c][i][j] = 1.0f / (1.0f +  (float)Math.exp(-x[n][c][i][j]));
					}
				}
			}
		}
		return r;
	}

	@Override
	public float[][][][] diff(float[][][][] x) {
		// TODO Auto-generated method stub
		float[][][][] r = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						float tmp = (float) (1d / (1d + Math.exp(-x[n][c][i][j])));
						r[n][c][i][j] = tmp * (1.0f - tmp);
					}
				}
			}
		}
		return r;
	}

	@Override
	public float[][][] diff2d() {
		// TODO Auto-generated method stub
		this.diff2d = MatrixUtils.clone(this.active(this.input2d));
		for(int c = 0;c<this.output2d.length;c++) {
			for(int i = 0;i<this.output2d[c].length;i++) {
				for(int j = 0;j<this.output2d[c][i].length;j++) {
					this.diff2d[c][i][j] = this.output2d[c][i][j] * (1.0f - this.output2d[c][i][j]);
				}
			}
		}
		return this.diff2d;
	}

	@Override
	public float[][][] activeTemp(float[][][] x) {
		// TODO Auto-generated method stub
		float[][][] output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					output2d[c][i][j] = (float) (1d / (1d + Math.exp(-x[c][i][j])));
				}
			}
		}
		return output2d;
	}

	@Override
	public float[][][] diffTemp(float[][][] x) {
		// TODO Auto-generated method stub
		float[][][] diff2d = MatrixUtils.clone(this.active(x));
		for(int c = 0;c<diff2d.length;c++) {
			for(int i = 0;i<diff2d[c].length;i++) {
				for(int j = 0;j<diff2d[c][i].length;j++) {
					diff2d[c][i][j] = diff2d[c][i][j] * (1.0f - diff2d[c][i][j]);
				}
			}
		}
		return this.diff2d;
	}

	public static void main(String[] args) {
		Sigmoid function = new Sigmoid();
		float[] x = new float[] {-0.033103351676417064f,1.0112810799899228f,0.0f,0.12f};
		function.input = x;
		float error = function.gradientCheck(x);
		System.out.println("error:"+error);
		System.out.println(JsonUtils.toJson(function));
	}

}
