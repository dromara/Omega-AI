package com.omega.engine.nn.layer.clip.bert;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * BertIntermediate Layer
 * @author Administrator
 *
 */
public class BertIntermediateLayer extends Layer{
	
	public FullyLayer linear;
	private GeluLayer active;

	public BertIntermediateLayer(int hiddenSize,int intermediateSize) {
		this.channel = 1;
		this.height = 1;
		this.width = hiddenSize;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = intermediateSize;
		this.initLayers();
	}
	
	public BertIntermediateLayer(int hiddenSize,int intermediateSize,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.channel = 1;
		this.height = 1;
		this.width = hiddenSize;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = intermediateSize;
		this.initLayers();
	}
	
	public void initLayers() {

		this.linear = new FullyLayer(width, oWidth, true, network);

		this.active = new GeluLayer(linear, true);
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
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
		
		linear.forward(input);
		
		active.forwardOld(linear.getOutput());
		
		this.output = active.getOutput();

	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		active.back(delta);
		linear.back(active.diff);
		this.diff = linear.diff;
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
		linear.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.bert_inter;
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		linear.saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		linear.loadModel(inputStream);
	}
	
	public static void main(String[] args) {
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		linear.accGrad(scale);
	}

}
