package com.omega.engine.nn.layer.unet;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * gn + activefunction + conv
 * @author Administrator
 *
 */
public class UNetResnetBlockLayer2 extends Layer{
	
	private int groups = 32;
	
	private int timeDim;
	
	public GNLayer norm1;
	
	public GNLayer norm2;
	
	private SiLULayer act1;
	
	private SiLULayer act2;
	
	public ConvolutionLayer conv1;
	
	public ConvolutionLayer conv2;
	
	public ConvolutionLayer residual;
	
	public UNetTEmbLayer temb;
	
	private Tensor tout;
	
	private Tensor dt;
	
	public UNetResnetBlockLayer2(int channel,int oChannel,int height,int width,int timeDim,int groups, Network network) {
		this.network = network;
		this.groups = groups;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		this.timeDim = timeDim;
		initLayers();
	}
	
	public void initLayers() {
		
		if(channel != oChannel) {
			residual = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, true, this.network);
			residual.setUpdater(UpdaterFactory.create(this.network));
			residual.paramsInit = ParamsInit.silu;
		}

		norm1 = new GNLayer(groups, channel, height, width, BNType.conv_bn, this);
		
		act1 = new SiLULayer(norm1);
		
		conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, this.network);
		conv1.setUpdater(UpdaterFactory.create(this.network));
		conv1.paramsInit = ParamsInit.silu;
		
		temb = new UNetTEmbLayer(timeDim, oChannel, network);
		
		norm2 = new GNLayer(groups, conv1);
		
		act2 = new SiLULayer(norm2);
		
		conv2 = new ConvolutionLayer(oChannel, oChannel, width, height, 3, 3, 1, 1, true, this.network);
		conv2.setUpdater(UpdaterFactory.create(this.network));
		conv2.paramsInit = ParamsInit.silu;
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(tout == null || tout.number != this.number) {
			tout = Tensor.createGPUTensor(tout, number, oChannel, height, width, true);
		}
		
		if(output == null || output.number != this.number) {
			output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(tout == null || tout.number != this.number) {
			tout = Tensor.createGPUTensor(tout, number, oChannel, height, width, true);
		}
		
		if(output == null || output.number != this.number) {
			output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(dt == null || dt.number != this.number) {
			dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
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

	public void output(Tensor t) {
		// TODO Auto-generated method stub

		Tensor x = input;

		if(channel != oChannel) {
			residual.forward(input);
			x = residual.getOutput();
		}

		norm1.forward(input);

		act1.forward(norm1.getOutput());
		conv1.forward(act1.getOutput());

		temb.forward(t);
		
		Tensor_OP().add(conv1.getOutput(), temb.getOutput(), tout, height * width);

		norm2.forward(tout);
		act2.forward(norm2.getOutput());
		conv2.forward(act2.getOutput());
		
		Tensor_OP().add(conv2.getOutput(), x, output);

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
	
	public void diff(Tensor tDiff) {
		// TODO Auto-generated method stub
		
		conv2.back(delta);
		act2.back(conv2.diff);
		norm2.back(act2.diff);
		
		dt.clearGPU();
		Tensor_OP().sum(norm2.diff, dt, 2);
		temb.back(dt);
		Tensor_OP().add(tDiff, temb.diff, tDiff);
		
		conv1.back(norm2.diff);
		act1.back(conv1.diff);
		norm1.back(act1.diff);
		
		Tensor d = delta;
		
		if(channel != oChannel) {
			residual.back(delta);
			d = residual.diff;
		}
		
		Tensor_OP().add(d, norm1.diff, norm1.diff);
		
		this.diff = norm1.diff;
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
	
	public void forward(Tensor input,Tensor t) {
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
		this.output(t);
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
	
	public void back(Tensor delta,Tensor tDiff) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(tDiff);

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub

		norm1.update();
		conv1.update();
		
		temb.update();
		
		norm2.update();
		conv2.update();
		
		if(channel != oChannel) {
//			System.err.println(residual.diffW);
			residual.update();
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
		if(channel != oChannel) {
			residual.accGrad(scale);
		}
		norm1.accGrad(scale);
		conv1.accGrad(scale);
		
		temb.accGrad(scale);
		
		norm2.accGrad(scale);
		conv2.accGrad(scale);
	}

}
