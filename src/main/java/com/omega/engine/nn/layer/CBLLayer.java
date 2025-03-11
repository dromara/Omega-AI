package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * conv + bn + activefunction
 * @author Administrator
 *
 */
public class CBLLayer extends Layer{

	private int kHeight = 3;
	
	private int kWidth = 3;
	
	private int padding = 1;
	
	private int stride = 2;
	
	private ConvolutionLayer convLayer;
	
	private BNLayer bnLayer;
	
	private ActiveFunctionLayer activeLayer;
	
	private ActiveType activeType;
	
	public CBLLayer(int channel,int oChannel,int height,int width,int kHeight,int kWidth,int stride,int padding, ActiveType activeType, Network network) {
		this.network = network;
		this.activeType = activeType;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.kHeight = kHeight;
		this.kWidth = kWidth;
		this.stride = stride;
		this.padding = padding;
		this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
		this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
		initLayers();
	}
	
	public CBLLayer(int channel,int oChannel,int height,int width,int kHeight,int kWidth,int stride,int padding, String activeType, Network network) {
		this.network = network;
		this.activeType = ActiveType.valueOf(activeType);
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.kHeight = kHeight;
		this.kWidth = kWidth;
		this.stride = stride;
		this.padding = padding;
		this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
		this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
		initLayers();
	}

	public void initLayers() {
		
		convLayer = new ConvolutionLayer(channel, oChannel, width, height, kHeight, kWidth, padding, stride, false, this.network, activeType);
		convLayer.setUpdater(UpdaterFactory.create(this.network));
		
		bnLayer = new BNLayer(convLayer);
		
//		System.out.println("bnLayer.preLayer:"+bnLayer.preLayer);
		
		switch (activeType) {
		case sigmoid:
			activeLayer = new SigmodLayer(bnLayer);
			break;
		case relu:
			activeLayer = new ReluLayer(bnLayer);
			break;
		case leaky_relu:
			activeLayer = new LeakyReluLayer(bnLayer);
			break;
		case tanh:
			activeLayer = new TanhLayer(bnLayer);
			break;
		case silu:
			activeLayer = new SiLULayer(bnLayer);
			break;
		default:
			throw new RuntimeException("The cbl layer is not support the ["+activeType+"] active function.");
		}
		
//		System.out.println("activeLayer.preLayer:"+activeLayer.preLayer);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.org_delta == null || output.number != org_delta.number){
			this.org_delta = Tensor.createTensor(org_delta, number, output.channel, output.height, output.width, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		convLayer.forward(this.input);
		bnLayer.forward(convLayer.output);
		activeLayer.forward(bnLayer.output);
		this.output = activeLayer.output;
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		baseKernel().copy_gpu(delta, this.org_delta, delta.getDataLength(), 1, 1);
		activeLayer.back(this.org_delta);
		bnLayer.back(activeLayer.diff);
		convLayer.back(bnLayer.diff);
		this.diff = convLayer.diff;
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
		convLayer.update();
		bnLayer.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.cbl;
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
	
	public ConvolutionLayer getConvLayer() {
		return convLayer;
	}

	public BNLayer getBnLayer() {
		return bnLayer;
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		convLayer.accGrad(scale);
		bnLayer.accGrad(scale);
	}

}
