package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveFunction;
import com.omega.engine.nn.data.Blob;

/**
 * Recurrent Layer
 * @author Administrator
 *
 */
public class RecurrentLayer extends Layer{
	
	private float[][] rWeight;
	
	private ActiveFunction activeFunction;
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.weight = RandomUtils.xavierRandom(this.width, this.oWidth, this.width, this.oWidth);
		this.bias = MatrixUtils.zero(this.oWidth);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		if(this.input != null) {

			for(int m = 0;m<this.number;m++) {
				for(int w = 0;w<oWidth;w++) {
					for(int i = 0;i<width;i++) {
						this.output.maxtir[m][0][0][w] += input.maxtir[m][0][0][i] * weight[i][w] + this.getPreValue(m, 0, 0, w) * rWeight[i][w];
					}
					if(hasBias) {
						this.output.maxtir[m][0][0][w] += bias[w];
					}
				}
			}
			
			if(this.activeFunction != null) {
				this.output.maxtir = this.activeFunction.active(this.output.maxtir);
			}

		}
		
	}
	
	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 计算输出
		 */
		this.output();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	public float[][] getrWeight() {
		return rWeight;
	}

	public void setrWeight(float[][] rWeight) {
		this.rWeight = rWeight;
	}
	
	public float getPreValue(int n,int b,int y,int x) {
		
		if((n - 1) < 0) {
			return 0;
		}else {
			return this.output.maxtir[n - 1][b][y][x];
		}
		
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}
	
}
