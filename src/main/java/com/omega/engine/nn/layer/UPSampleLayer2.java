package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.gpu.UpSampleKernel2;
import com.omega.engine.nn.network.Network;

/**
 * 上采用层
 * @author Administrator
 *
 */
public class UPSampleLayer2 extends Layer {
	
	private int scale = 2;
	
	private int ndim = 3;
	
	private UpSampleKernel2 kernel;
	
	public UPSampleLayer2(int channel,int height,int width,int scale) {
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.oChannel = channel;
		this.scale = scale;
		if(height > 1 && width > 1) {
			ndim = 4;
		}
		this.oHeight = this.height * scale;
		this.oWidth = this.width * scale;
	}
	
	public UPSampleLayer2(int channel,int height,int width,int scale,Network network) {
		this.network = network;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.oChannel = channel;
		this.scale = scale;
		if(height > 1 && width > 1) {
			ndim = 4;
		}
		this.oHeight = this.height * scale;
		this.oWidth = this.width * scale;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.output.number != number) {
			this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}

		if(kernel == null) {
			kernel = new UpSampleKernel2(this.scale, ndim, cuda());
		}
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.diff.number != number) {
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
		return LayerType.upsample;
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
