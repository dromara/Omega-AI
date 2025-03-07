package com.omega.engine.nn.layer.unet;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * UNetFFNBlockLayer
 * @author Administrator
 *
 */
public class UNetSpatialTransformerLayer extends Layer{
	
	private boolean seft_attn = false;
	
	private boolean cross_attn = false;
	
	private int contextDim;
	
	private int kvTime;

	private int headNum;
	
	public UNetTransformerBlockLayer transformer;
	
	public FullyLayer contextProj;
	
	private Tensor xt;
	
	private Tensor kvDiff;
	
	public UNetSpatialTransformerLayer(int channel,int height,int width,int headNum,int kvTime,int contextDim, boolean seft_attn, boolean cross_attn, Network network) {
		this.network = network;
		this.seft_attn = seft_attn;
		this.cross_attn = cross_attn;
		this.headNum = headNum;
		this.kvTime = kvTime;
		this.contextDim = contextDim;
		this.channel = channel;
		this.oChannel = channel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		initLayers();
	}
	
	public void initLayers() {

		if(cross_attn) {
			contextProj = new FullyLayer(contextDim, channel, true, network);
		}

		transformer = new UNetTransformerBlockLayer(channel, headNum, height * width, kvTime, channel, seft_attn, cross_attn, network);
		
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(this.xt == null || this.xt.number != this.number) {
			this.xt = Tensor.createGPUTensor(this.xt, number, height, width, channel, true);
		}else {
			xt.viewOrg();
		}
		if(this.output == null || this.output.number != this.number) {
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(kvDiff == null || kvDiff.number * kvTime != this.number * kvTime) {
			kvDiff = Tensor.createGPUTensor(kvDiff, this.number * kvTime, 1, 1, channel, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

	}
	
	public void output(Tensor context) {
		// TODO Auto-generated method stub
		//[b, c, h, w] --> [b, h*w, c]
		
		Tensor_OP().permute(input, xt, new int[] {0, 2, 3, 1});
		
		xt.view(number, height * width, 1, channel);
		
		Tensor cx = null;
		
		if(cross_attn) {
			contextProj.forward(context);
			cx = contextProj.getOutput();
//			cx.showDM("cx");
		}
		
		transformer.forward(xt, cx);
//		transformer.getOutput().showShape("tos");
		//[b, h*w, c]--> [b, c, h, w]
		this.output.view(number, channel, 1, height * width);
//		transformer.getOutput().showDM("to");

		Tensor_OP().permute(transformer.getOutput(), this.output, new int[] {0, 3, 2, 1});

		this.output.viewOrg();
		
//		this.output.showDM("to1111");
//		this.input.showDM("input1111");
		Tensor_OP().add(this.output, this.input, this.output);
		
//		this.output.showDM("to2222");
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		//[b, c, h, w] --> [b, h*w, c]
		Tensor deltaT = this.output.view(number, height, width, channel);
		Tensor_OP().permute(delta, deltaT, new int[] {0, 2, 3, 1});
		deltaT.view(number, height * width, 1, channel);
		
		transformer.back(deltaT, kvDiff);
		
		if(cross_attn) {
			contextProj.back(kvDiff);
		}
		
		this.diff = this.output.view(number, channel, 1, height * width);
		Tensor_OP().permute(transformer.diff, this.diff, new int[] {0, 3, 2, 1});
		this.diff.view(number, channel, height, width);
		Tensor_OP().add(delta, this.diff, this.diff);
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
	
	public void forward(Tensor input,Tensor context) {
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
		this.output(context);
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
		transformer.update();
		if(cross_attn) {
			contextProj.update();
		}
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
		transformer.accGrad(scale);
		if(cross_attn) {
			contextProj.accGrad(scale);
		}
	}

}
