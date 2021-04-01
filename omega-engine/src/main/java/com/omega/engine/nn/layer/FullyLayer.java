package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.active.ActiveFunction;
import com.omega.engine.updater.Updater;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{
	
	public FullyLayer(int index,int inputNum,int outputNum) {
		this.index = index;
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.layerType = LayerType.full;
		this.initParam();
	}
	
	public FullyLayer(int inputNum,int outputNum,ActiveFunction activeFunction) {
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.activeFunction = activeFunction;
		this.layerType = LayerType.full;
		this.initParam();
	}
	
	public FullyLayer(int index,int inputNum,int outputNum,ActiveFunction activeFunction) {
		this.index = index;
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.activeFunction = activeFunction;
		this.layerType = LayerType.full;
		this.initParam();
	}
	
	public FullyLayer(int inputNum,int outputNum,ActiveFunction activeFunction,Updater updater) {
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.activeFunction = activeFunction;
		this.updater = updater;
		this.layerType = LayerType.full;
		this.initParam();
	}
	
	public FullyLayer(int index,int inputNum,int outputNum,ActiveFunction activeFunction,Updater updater) {
		this.index = index;
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.activeFunction = activeFunction;
		this.updater = updater;
		this.layerType = LayerType.full;
		this.initParam();
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.zero(this.outputNum);
		this.diff = MatrixOperation.zero(this.inputNum);
		this.nextDiff = MatrixOperation.zero(this.outputNum);
		this.delta = MatrixOperation.zero(this.outputNum);
		this.deltaW = MatrixOperation.zero(this.inputNum,this.outputNum);
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.weight = MatrixOperation.gaussianRandom(this.inputNum, this.outputNum, 0.1);
		this.bias = MatrixOperation.zero(this.outputNum);
//		this.bias = MatrixOperation.gaussianRandom(this.outputNum, 0.1);
	}

	public void input(double[] data) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(data);
	}

	public void nextDiff(double[] data) {
		// TODO Auto-generated method stub
		this.nextDiff = MatrixOperation.clone(data);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		if(this.input != null) {
			for(int o = 0;o<this.outputNum;o++) {
				for(int i = 0;i<this.inputNum;i++) {
					this.output[o] += this.input[i] * this.weight[i][o];
				}
				this.output[o] += this.bias[o];
			}
		}
	}
	
	@Override
	public void active() {
		// TODO Auto-generated method stub
		if(this.activeFunction != null) {
//			System.out.println("realOuput:"+JsonUtils.toJson(this.output));
//			System.out.println("realActive:"+JsonUtils.toJson(this.activeFunction.active(this.output)));
			this.active = this.activeFunction.active(this.output);
		}else{
			this.active = this.output;
		}

	}
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		if(this.activeFunction!=null) {
			this.activeFunction.diff();
//			System.out.println("nextDiff:"+JsonUtils.toJson(this.nextDiff));
//			System.out.println("activeFunction.diff:"+JsonUtils.toJson(this.activeFunction.diff));
			/**
			 * 计算当前层delta
			 * delta(i) = diff(i + 1) * f'(xi)
			 */
			this.delta = MatrixOperation.multiplication(this.nextDiff, this.activeFunction.diff);
			
		}else {

			/**
			 * 计算当前层delta
			 * delta(i) = diff(i + 1)
			 */
			this.delta = this.nextDiff;
			
		}
//		System.out.println(this.index+":activieDiff:"+JsonUtils.toJson(this.activeFunction.diff));
//		System.out.println(this.index+":delta:"+JsonUtils.toJson(this.delta));
		/**
		 * 计算当前层weight梯度
		 * deltaW(i) = delta(i) * input(i)
		 */
		for(int i = 0;i<this.deltaW.length;i++) {
			for(int j = 0;j<this.deltaW[i].length;j++) {
				this.deltaW[i][j] = this.delta[j] * this.input[i];
			}
		}
		
		/**
		 * 计算当前层误差
		 */
		for(int i = 0;i<this.inputNum;i++) {
			for(int j = 0;j<this.outputNum;j++) {
				this.diff[i] += this.delta[j] * this.weight[i][j];
			}
		}
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		this.output();
		this.active();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.diff();
		if(this.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	/**
	 * w(t) = w(t-1) + θ * deltaW
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
			for(int i = 0;i<this.outputNum;i++) {
				this.bias[i] -= this.learnRate * this.delta[i];
			}
		}
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.full;
	}

	@Override
	public double[] activeTemp(double[] output) {
		// TODO Auto-generated method stub
		return this.activeFunction.activeTemp(output);
	}

	@Override
	public double[] diffTemp() {
		// TODO Auto-generated method stub
		return this.activeFunction.diff;
	}

	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

}
