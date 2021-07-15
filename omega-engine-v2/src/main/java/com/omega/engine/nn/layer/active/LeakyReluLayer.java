package com.omega.engine.nn.layer.active;

import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.layer.LayerType;

/**
 * Relu active function Layer
 * @author Administrator
 *
 */
public class LeakyReluLayer extends ActiveFunctionLayer {
	
	private double leak = 0.2d;
	
	public LeakyReluLayer() {
		
	}
	
	public LeakyReluLayer(double leak) {
		this.leak = leak;
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<this.number;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<channel;c++) {
						for(int h = 0;h<height;h++) {
							for(int w = 0;w<width;w++) {
								if(input.maxtir[index][c][h][w] >= 0) {
									output.maxtir[index][c][h][w] = input.maxtir[index][c][h][w];
								}else {
									output.maxtir[index][c][h][w] = input.maxtir[index][c][h][w] * leak;
								}
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
		
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		for(int n = 0;n<this.number;n++) {
			for(int c = 0;c<this.channel;c++) {
				for(int h = 0;h<this.height;h++) {
					for(int w = 0;w<this.width;w++) {
						if(this.input.maxtir[n][c][h][w] >= 0) {
							this.diff.maxtir[n][c][h][w] = this.delta.maxtir[n][c][h][w];
						}else {
							this.diff.maxtir[n][c][h][w] = this.delta.maxtir[n][c][h][w] * leak;
						}
					}
				}
			}
		}
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
		return LayerType.relu;
	}

	@Override
	public double[][][][] output(double[][][][] input) {
		// TODO Auto-generated method stub
		
		double[][][][] output = new double[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<this.number;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<channel;c++) {
						for(int h = 0;h<height;h++) {
							for(int w = 0;w<width;w++) {
								if(input[index][c][h][w] > 0) {
									output[index][c][h][w] = input[index][c][h][w];
								}else {
									output[index][c][h][w] = leak * input[index][c][h][w];
								}
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
		
		return output;
	}

}
