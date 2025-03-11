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
public class UNetResnetBlockLayer extends Layer{
	
	private int groups = 32;
	
	public GNLayer norm;
	
	private SiLULayer act;
	
	public ConvolutionLayer conv;
	
	public UNetResnetBlockLayer(int channel,int oChannel,int height,int width,int groups, Network network) {
		this.network = network;
		this.groups = groups;
		this.channel = channel;
		this.oChannel = channel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		initLayers(oChannel);
	}
	
	public void initLayers(int oc) {
		
		//int groupNum,int channel,int height,int width,BNType bnType,Layer preLayer
		norm = new GNLayer(groups, channel, height, width, BNType.conv_bn, this);
		
		act = new SiLULayer(norm);
		
		conv = new ConvolutionLayer(channel, oc, width, height, 3, 3, 1, 1, true, this.network);
		conv.setUpdater(UpdaterFactory.create(this.network));
		conv.paramsInit = ParamsInit.silu;
		
		this.oChannel = oc;
		this.oHeight = conv.oHeight;
		this.oWidth = conv.oWidth;

	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
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
//		input.showShape();
		norm.forward(input);
//		norm.getOutput().showShape();
//		norm.gamma.showDM();
//		norm.beta.showDM();
//		norm.getOutput().showDM("norm");
		act.forward(norm.getOutput());
		
		conv.forward(act.getOutput());

		this.output = conv.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

		conv.back(delta);
//		conv.diff.showDM("afout");
		act.back(conv.diff);
//		act.diff.showDM("nfout");
		norm.back(act.diff);
		
		this.diff = norm.diff;
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
		conv.update();
		norm.update();
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
		conv.accGrad(scale);
		norm.accGrad(scale);
	}

}
