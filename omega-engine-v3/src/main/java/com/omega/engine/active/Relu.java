package com.omega.engine.active;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;

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
	public float[] active(float[] x) {
		// TODO Auto-generated method stub
		this.input = MatrixUtils.clone(x);
		this.output = MatrixUtils.zero(x.length);
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
	public float[] diff() {
		// TODO Auto-generated method stub
		this.diff = MatrixUtils.zero(this.input.length);
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
	public float[] activeTemp(float[] x) {
		// TODO Auto-generated method stub
		float[] output = MatrixUtils.zero(x.length);
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
	public float[] diffTemp(float[] x) {
		// TODO Auto-generated method stub
		float[] diff = MatrixUtils.zero(x.length);
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
	public float[][][] active(float[][][] x) {
		// TODO Auto-generated method stub
		this.input2d = MatrixUtils.clone(x);
		this.output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
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
	public float[][][][] active(float[][][][] x) {
		// TODO Auto-generated method stub
		float[][][][] r = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						if(x[n][c][i][j] > 0) {
							r[n][c][i][j] = x[n][c][i][j];
						}else {
							r[n][c][i][j] = 0;
						}
					}
				}
			}
		}
		return r;
	}
	
	@Override
	public float[][][] diff2d() {
		// TODO Auto-generated method stub
		this.diff2d = MatrixUtils.zero(this.input2d.length,this.input2d[0].length,this.input2d[0][0].length);
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
	public float[][][][] diff(float[][][][] x) {
		// TODO Auto-generated method stub
		float[][][][] r = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						if(x[n][c][i][j] > 0) {
							r[n][c][i][j] = 1;
						}else {
							r[n][c][i][j] = 0;
						}
					}
				}
			}
		}
		return r;
	}
	
	@Override
	public float[][][] activeTemp(float[][][] x) {
		// TODO Auto-generated method stub
		float[][][] output2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
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
	public float[][][] diffTemp(float[][][] x) {
		// TODO Auto-generated method stub
		float[][][] diff2d = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
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
		float[] x = new float[] {0.1f,-0.03f,0.25f,0.4f,-0.87f,0.12f,-0.001f,0.0f};
		float error = function.gradientCheck(x);
		System.out.println("error:"+error);
		System.out.println(JsonUtils.toJson(function));
	}

}
