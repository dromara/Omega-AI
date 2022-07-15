package com.omega.engine.nn.layer;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class BasicBlockLayer extends Layer {
	
	private Layer identity;
	
	private int kHeight = 3;
	
	private int kWidth = 3;
	
	private int padding = 1;
	
	private boolean downSample = false;
	
	private int fisrtLayerStride = 2;
	
	public BasicBlockLayer(int channel,int oChannel,int height,int width,boolean downSample) {
		this.layers = new ArrayList<Layer>();
		
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.downSample = downSample;
		
		if(this.downSample) {
			this.oHeight = (height + padding * 2 - kHeight) / fisrtLayerStride + 1;
			this.oWidth = (width + padding * 2 - kWidth) / fisrtLayerStride + 1;
		}else {
			this.oHeight = height;
			this.oWidth = width;
		}

		initLayer();
		
	}
	
	public void initLayer() {
		
		ConvolutionLayer conv1 = null;
		
		if(downSample) {
			this.setIdentity(new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, fisrtLayerStride, false));
			this.getIdentity().index = 0;
			this.getIdentity().isIdentity = true;
			this.getIdentity().parent = this;
			conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, fisrtLayerStride, false);
		}else {
			conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, false);
		}

		BNLayer bn1 = new BNLayer();
		
		ReluLayer a1 = new ReluLayer();
		
		ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
		
		BNLayer bn = new BNLayer();
		
		this.addLayer(conv1);
		this.addLayer(bn1);
		this.addLayer(a1);
		this.addLayer(conv2);
		this.addLayer(bn);
		
	}
	
	public void addLayer(Layer layer) {
		
		layer.setIndex(this.layers.size());
		
		layer.parent = this;
		
		this.layers.add(layer);
		
	}
	
	@Override
	public void init() {
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
	}
	
	@Override
	public void initBack() {
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		float[][][][] x = this.input.maxtir;
		
		if(getIdentity() != null) {
			getIdentity().forward();
			x = getIdentity().output.maxtir;
		}
		
		/**
		 * o = x + f(x)
		 */
		for(Layer layer:layers) {
			layer.forward();
		}
		
		float[][][][] x2 = layers.get(layers.size() - 1).output.maxtir;
		
		this.output.maxtir = MatrixOperation.add(x, x2);
		
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * deltax = deltao * (f'(x) + 1)
		 */
		for(int i = layers.size() - 1;i>=0;i--) {
			Layer layer = layers.get(i);
			layer.back();
		}
		
		float[][][][] delta2 = layers.get(0).diff.maxtir;
		
		if(getIdentity() != null) {
			getIdentity().back();
			this.diff.maxtir = MatrixOperation.add(delta2, getIdentity().diff.maxtir);
		}else {
			this.diff.maxtir = MatrixOperation.add(delta2, this.delta.maxtir);
		}

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
		
		for(Layer layer:layers) {
			layer.update();
		}
		
		if(this.identity != null) {
			this.identity.update();
		}
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.block;
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

	public List<Layer> getLayers() {
		return layers;
	}

	public Layer getIdentity() {
		return identity;
	}

	public void setIdentity(Layer identity) {
		this.identity = identity;
	}

}
