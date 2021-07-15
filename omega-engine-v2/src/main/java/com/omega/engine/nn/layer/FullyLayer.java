package com.omega.engine.nn.layer;

import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{

	public FullyLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.initParam();
	}

	public FullyLayer(int inputNum,int outputNum,boolean hasBias) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.initParam();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
		this.deltaB = MatrixUtils.zero(this.oWidth);
		this.deltaW = MatrixUtils.zero(this.width,this.oWidth);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
//		this.weight = MatrixOperation.heRandom(this.width, this.oWidth, this.width);
//		this.weight = MatrixOperation.gaussianRandom(this.width, this.oWidth, 0.01);
		this.weight = RandomUtils.xavierRandom(this.width, this.oWidth, this.width, this.oWidth);
		this.bias = MatrixUtils.zero(this.oWidth);
//		this.bias = MatrixOperation.gaussianRandom(this.outputNum, 0.1);
	}
	
	
//
//	@Override
//	public void output() {
//		// TODO Auto-generated method stub
//
//		if(this.input != null) {
//			for(int m = 0;m<this.number;m++) {
//				for(int w = 0;w<this.oWidth;w++) {
//					for(int i = 0;i<this.width;i++) {
//						this.output.maxtir[m][0][0][w] += this.input.maxtir[m][0][0][i] * this.weight[i][w];
//					}
//					if(hasBias) {
//						this.output.maxtir[m][0][0][w] += this.bias[w];
//					}
//				}
//			}
//		}
//
////		System.out.println(JsonUtils.toJson(this.output));
//		
//	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {

			Vector<Task<Object>> workers = new Vector<Task<Object>>();

			for(int m = 0;m<this.number;m++) {
				final int index = m;
				workers.add(new Task<Object>(index) {
					@Override
				    public Object call() throws Exception {
						for(int w = 0;w<oWidth;w++) {
							for(int i = 0;i<width;i++) {
								output.maxtir[index][0][0][w] += input.maxtir[index][0][0][i] * weight[i][w];
							}
							if(hasBias) {
								output.maxtir[index][0][0][w] += bias[w];
							}
						}
						return null;
					}
				});
			}

			TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
			
		}
		
	}
	
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * 计算当前层weight梯度
		 * deltaW(i) = 1/m * ∑ delta(j) * input(i)
		 * 计算deltaW平均值
		 */
		for(int m = 0;m<this.number;m++) {
			for(int i = 0;i<this.deltaW.length;i++) {
				for(int j = 0;j<this.deltaW[i].length;j++) {
					this.deltaW[i][j] += this.delta.maxtir[m][0][0][j] * this.input.maxtir[m][0][0][i] / this.number;
				}
			}
		}
		
		if(hasBias) {

			/**
			 * 计算当前层weight梯度
			 * deltaB(i) = 1/m * ∑ delta(i) * input(i)
			 * 计算deltaB平均值
			 */
			for(int m = 0;m<this.number;m++) {
				for(int i = 0;i<this.deltaB.length;i++) {
					this.deltaB[i] += this.delta.maxtir[m][0][0][i] / this.number;
				}
			}

		}
		
		/**
		 * 计算当前层误差
		 */
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		for(int n = 0;n<this.number;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int i = 0;i<width;i++) {
						for(int j = 0;j<oWidth;j++) {
							diff.maxtir[index][0][0][i] += delta.maxtir[index][0][0][j] * weight[i][j];
						}
					}
					return null;
				}
			});
		}
		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
		
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

	/**
	 * w(t) = w(t-1) + θ * deltaW
	 * b(t) = b(t-1) + θ * deltaB
	 * θ : learningRate
	 */
	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		if(this.updater != null){
			this.updater.update(this);
		}else{
			for(int i = 0;i<this.weight.length;i++) {
				for(int j = 0;j<this.weight[i].length;j++) {
					this.weight[i][j] -= this.learnRate * this.deltaW[i][j];
				}
			}
			for(int i = 0;i<this.oWidth;i++) {
				this.bias[i] -= this.learnRate * this.deltaB[i];
			}
		}
		
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("fully layer["+this.index+"]diff start:");
//		
//		MatrixOperation.printImage(this.active);
//		
//		System.out.println("fully layer["+this.index+"]diff end.");
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.full;
	}

	@Override
	public double[][][][] output(double[][][][] input) {
		// TODO Auto-generated method stub
		
		double[][][][] output = new double[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();

		for(int m = 0;m<this.number;m++) {
			final int index = m;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int w = 0;w<oWidth;w++) {
						for(int i = 0;i<width;i++) {
							output[index][0][0][w] += input[index][0][0][i] * weight[i][w];
						}
						if(hasBias) {
							output[index][0][0][w] += bias[w];
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
