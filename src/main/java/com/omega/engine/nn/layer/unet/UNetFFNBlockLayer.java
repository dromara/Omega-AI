package com.omega.engine.nn.layer.unet;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;

/**
 * UNetFFNBlockLayer
 * @author Administrator
 *
 */
public class UNetFFNBlockLayer extends Layer{
	
	private int mult = 4;
	
	public LNLayer norm;
	
	public FullyLayer linear1;
	
	private GeluLayer act;
	
	public FullyLayer linear2;
	
	public UNetFFNBlockLayer(int channel,int height,int width,int mult, Network network) {
		this.network = network;
		this.mult = mult;
		this.channel = channel;
		this.oChannel = channel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		initLayers();
	}
	
	public void initLayers() {

		norm = new LNLayer(this, BNType.fully_bn, 1, 1, width);
		
		//int groupNum,int channel,int height,int width,BNType bnType,Layer preLayer
		int iw = width;
		int ow = mult * iw;
		linear1 = new FullyLayer(iw, ow, true, network);
		
		act = new GeluLayer(linear1);
		
		linear2 = new FullyLayer(ow, iw, true, network);
		
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(this.output == null || this.output.number != this.number) {
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

//		input.showShape("norm-input");
		
		norm.forward(input);

//		norm.getOutput().showShape("norm-out");
		
		norm.getOutput().view(number * channel, 1, 1, width);
		
		linear1.forward(norm.getOutput());

		act.forward(linear1.getOutput());
		
		linear2.forward(act.getOutput());
		
		this.output.view(number * channel, 1, 1, width);
		
		Tensor_OP().add(input, linear2.getOutput(), this.output);
		
		norm.getOutput().viewOrg();
		
		this.output.viewOrg();
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		delta.view(number * channel, 1, 1, width);
		
		linear2.back(delta);

		act.back(linear2.diff);

		linear1.back(act.diff);
		
		norm.back(linear1.diff);
		
		Tensor_OP().add(norm.diff, delta, delta);
		
		delta.viewOrg();
		
		this.diff = delta;
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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
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
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		norm.update();
		linear1.update();
		linear2.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.unet_down;
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
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		norm.accGrad(scale);
		linear1.accGrad(scale);
		linear2.accGrad(scale);
	}

}
