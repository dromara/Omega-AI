package com.omega.engine.nn.layer.transformer;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;

/**
 * RoPE Layer
 * @author Administrator
 *
 */
public class RoPELayer extends Layer {
	
	public Layer preLayer;
	
	private RoPEKernel kernel;
	
	public RoPELayer() {
		
	}
	
	public RoPELayer(Layer preLayer) {
		this.setPreLayer(preLayer);
	}
	
	public RoPELayer(Network network) {
		this.network = network;
	}
	
	public void setPreLayer(Layer pre) {
		this.preLayer = pre;
		this.network = pre.network;
		this.channel = preLayer.oChannel;
		this.height = preLayer.oHeight;
		this.width = preLayer.oWidth;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		
		if(kernel == null) {
			kernel = new RoPEKernel(cuda());
		}
		this.number = this.network.number;
		initParam();
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.channel = input.channel;
		this.height = input.height;
		this.width = input.width;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
		
		if(kernel == null) {
			kernel = new RoPEKernel(cuda());
		}
		this.number = input.number;
		initParam();
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
		if(this.output == null || this.number != this.output.number) {
			this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}

	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number) {
			this.diff = Tensor.createTensor(this.diff, number, channel, height, width, true);
		}
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
	}
	
	public void output(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		kernel.forward(cos, sin,input, output);
//		kernel.forward32(input, output);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

	}
	
	public void diff(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		kernel.backward(cos, sin, delta, diff);
//		kernel.backward32(delta, diff);
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
		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();

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
		return LayerType.rope;
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
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
//		input.showDMByNumber(0);
		/**
		 * 参数初始化
		 */
		this.init(input);
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 计算输出
		 */
		this.output();
//		getOutput().showDMByNumber(0);
	}
	
	public void forward(Tensor cos,Tensor sin,Tensor input) {
		// TODO Auto-generated method stub
//		input.showDMByNumber(0);
		/**
		 * 参数初始化
		 */
		this.init(input);
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 计算输出
		 */
		this.output(cos, sin);
//		getOutput().showDMByNumber(0);
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
	
	public void back(Tensor cos,Tensor sin,Tensor delta) {
		// TODO Auto-generated method stub
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(cos, sin);
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
