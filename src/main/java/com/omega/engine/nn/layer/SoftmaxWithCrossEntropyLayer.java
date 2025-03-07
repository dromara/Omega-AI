package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.SoftmaxKernel;

/**
 * SoftmaxWithCrossEntropyLayer
 * 
 * @author Administrator
 * 
 * @remark
 * 
 * The softmax function support only cross entropy loss function now.
 * 
 */
public class SoftmaxWithCrossEntropyLayer extends Layer {
	
	private Tensor currentLabel;
	
	private SoftmaxKernel kernel;
	
	public SoftmaxWithCrossEntropyLayer(int inputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = inputNum;
		this.initParam();
		this.initKernel();
	}
	
	public void initKernel() {
		kernel = new SoftmaxKernel(cuda());
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(output == null || output.number != number) {
			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
			this.diff = new Tensor(number, channel, height, width, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		
//		input.showDM();
		
		kernel.softmax(input, output);
//		input.showDM();
//		output.showDM();
		
//		Tensor o = new Tensor(number, channel, height, width);
//		
//		input.syncHost();
//		
//		kernel.cpuForward(input, o);
		
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		kernel.backward2(output, currentLabel, diff);
//		diff.showDM();
//
//		System.out.println("softmax-diff:");
//		diff.showDM();
//		
//		Tensor d = new Tensor(number, channel, height, width);
//		
//		output.syncHost();
//		
//		kernel.cpuBackward(output, currentLabel, d);

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		
		this.init();
		
		/**
		 * 设置输入
		 */
		this.setInput();
		
//		System.out.println("====");
//		
//		MatrixOperation.printImage(this.input.maxtir[0]);
		
		this.output();

//		MatrixOperation.printImage(this.output.maxtir[0]);
//		output.showDM();
		
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
	}

	public void setCurrentLabel(Tensor currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax_cross_entropy;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}
	

	@Override
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(inpnut);
		/**
		 * 计算输出
		 */
		this.output();
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

}
