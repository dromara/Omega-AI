package com.omega.engine.nn.layer.transformer;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * PoswiseFeedForward Layer
 * @author Administrator
 *
 */
public class PoswiseFeedForwardLinearLayer extends Layer{
	
	private int embedDim = 0;
	
	private int nChannel = 1;
	
	private boolean bias = false;
	
	private boolean layer_norm = false;
	
	private FullyLayer linear1;
	private GeluLayer relu1;
	private FullyLayer linear2;

	private LNLayer lnLayer;
	
	private Tensor ro;

	public PoswiseFeedForwardLinearLayer(int embedDim,int nChannel,boolean bias,boolean layer_norm) {
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public PoswiseFeedForwardLinearLayer(int embedDim,int nChannel,boolean bias,boolean layer_norm,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);

		this.relu1 = new GeluLayer(linear1);
		
		this.linear2 = new FullyLayer(nChannel, embedDim, bias, network);

		if(this.layer_norm) {
			this.lnLayer = new LNLayer(this.linear2);
		}
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		if(this.ro == null || this.ro.number != this.number) {
			this.ro = Tensor.createTensor(this.ro, number, 1, 1, embedDim, true);
		} 
//		resize();
	}
	
	public void resize() {
		this.ro.viewOrg();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
//		if(this.cache_delta == null || output.number != cache_delta.number){
//			this.cache_delta = new Tensor(number, output.channel, output.height, output.width, true);
//		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		linear1.forward(input);
		
		relu1.forward(linear1.getOutput());

		linear2.forward(relu1.getOutput());
		
		Tensor_OP().add(linear2.getOutput(), this.input, this.ro);
		
		if(this.layer_norm) {
			this.lnLayer.forward(ro);
			this.output = this.lnLayer.getOutput();
		}else {
			this.output = ro;
		}

	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		if(this.layer_norm) {
			this.lnLayer.back(delta);
//			baseKernel.copy_gpu(delta, this.cache_delta, delta.getDataLength(), 1, 1);
			this.linear2.back(this.lnLayer.diff);
		}else {
			this.linear2.back(this.delta);
		}

		relu1.back(this.linear2.diff);
		
		linear1.back(relu1.diff);
		
		Tensor_OP().add(this.linear1.diff, this.lnLayer.diff, this.linear1.diff);

		this.diff = this.linear1.diff;
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 参数初始化
		 */
		this.init();
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
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 参数初始化
		 */
		this.init();
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
		linear1.update();
		linear2.update();
		if(layer_norm) {
			lnLayer.update();
		}
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.poswise_feed_forward;
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
	
	public static void main(String[] args) {
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		linear1.accGrad(scale);
		linear2.accGrad(scale);
		lnLayer.accGrad(scale);
	}

}
