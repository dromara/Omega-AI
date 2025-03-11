package com.omega.engine.nn.layer.vae.tiny;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class TinyVAEEncoder extends Layer {
	
	private int z_dims;
	
	public TinyVAEConvBlock block1;
	public TinyVAEConvBlock block2;
	public TinyVAEConvBlock block3;
	
	public ConvolutionLayer convOut;
	
	public TinyVAEEncoder(int channel,int height,int width, int z_dims, Network network) {
		this.network = network;
		this.z_dims = z_dims;
		this.channel = channel;
		this.oChannel = 256;
		this.height = height;
		this.width = width;
		
		initLayers();
		
		this.oHeight = block3.oHeight;
		this.oWidth = block3.oWidth;
	}
	
	public void initLayers() {
		block1 = new TinyVAEConvBlock(channel, 64, height, width, network);
		block2 = new TinyVAEConvBlock(64, 128, block1.oHeight, block1.oWidth, network);
		block3 = new TinyVAEConvBlock(128, 256, block2.oHeight, block2.oWidth, network);
		
		convOut = new ConvolutionLayer(256, z_dims, block3.oWidth, block3.oHeight, 1, 1, 0, 1, true, this.network);
		convOut.setUpdater(UpdaterFactory.create(this.network));
		convOut.paramsInit = ParamsInit.leaky_relu;
		
	}

	@Override
	public void init() {
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		block1.forward(this.input);
		block2.forward(block1.getOutput());
		block3.forward(block2.getOutput());
		
		convOut.forward(block3.getOutput());
		
		this.output = convOut.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		convOut.back(this.delta);
		
		block3.back(convOut.diff);
		block2.back(block3.diff);
		block1.back(block2.diff);

		this.diff = block1.diff;
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
		block1.update();
		block2.update();
		block3.update();
		
		convOut.update();
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

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub

		/**
		 * 参数初始化
		 */
		this.init();
		
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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
		block1.saveModel(outputStream);
		block2.saveModel(outputStream);
		block3.saveModel(outputStream);
		
		convOut.saveModel(outputStream);
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		
		block1.loadModel(inputStream);
		block2.loadModel(inputStream);
		block3.loadModel(inputStream);
		
		convOut.loadModel(inputStream);
		
	}

}
