package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.gpu.AdaptiveAvgPool2DKernel;
import com.omega.engine.nn.network.Network;

public class AdaptiveAvgPool2DLayer extends Layer {
	
	private AdaptiveAvgPool2DKernel kernel;
	
	public AdaptiveAvgPool2DLayer(int channel,int height,int width,int oHeight,int oWidth) {
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.oChannel = channel;
		this.oHeight = oHeight;
		this.oWidth = oWidth;
	}
	
	public AdaptiveAvgPool2DLayer(int channel,int height,int width,int oHeight,int oWidth,Network network) {
		this.network = network;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.oChannel = channel;
		this.oHeight = oHeight;
		this.oWidth = oWidth;
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.output.number != number) {
			this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
//			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}

		if(kernel == null) {
			kernel = new AdaptiveAvgPool2DKernel(cuda());
		}
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(diff == null || this.diff.number != number) {
			this.diff = new Tensor(number, channel, height, width, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		kernel.forward(input, output);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		kernel.backward(delta, diff);
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
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(input);
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
		return LayerType.avgpooling;
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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

}
