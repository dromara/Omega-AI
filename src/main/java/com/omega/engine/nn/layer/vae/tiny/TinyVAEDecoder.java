package com.omega.engine.nn.layer.vae.tiny;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class TinyVAEDecoder extends Layer {
	
	public ConvolutionTransposeLayer decoderInput;
	
	public TinyVAEConvTransposeBlock block1;
	public TinyVAEConvTransposeBlock block2;
	public TinyVAEConvTransposeBlock block3;
	
	private TanhLayer act;
	
	public TinyVAEDecoder(int channel,int oChannel,int height,int width, Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		
		initLayers();
		
		this.oHeight = block3.oHeight;
		this.oWidth = block3.oWidth;
	}
	
	public void initLayers() {
		decoderInput = new ConvolutionTransposeLayer(channel, 256, width, height, 1, 1, 0, 1, 1, 0, true, network);
		decoderInput.setUpdater(UpdaterFactory.create(this.network));
		decoderInput.paramsInit = ParamsInit.leaky_relu;
		block1 = new TinyVAEConvTransposeBlock(256, 128, decoderInput.oHeight, decoderInput.oWidth, network);
		block2 = new TinyVAEConvTransposeBlock(128, 64, block1.oHeight, block1.oWidth, network);
		block3 = new TinyVAEConvTransposeBlock(64, oChannel, block2.oHeight, block2.oWidth, network);
		
		act = new TanhLayer(block3);
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
		
		decoderInput.forward(this.input);
		
		block1.forward(decoderInput.getOutput());
		block2.forward(block1.getOutput());
		block3.forward(block2.getOutput());
		
//		this.output = block3.getOutput();
		
		act.forward(block3.getOutput());
		this.output = act.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		act.back(this.delta);
		
		block3.back(act.diff);
//		block3.back(this.delta);
		block2.back(block3.diff);
		block1.back(block2.diff);
		
		decoderInput.back(block1.diff);
		
		this.diff = decoderInput.diff;
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
		decoderInput.update();
		block1.update();
		block2.update();
		block3.update();
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
		
		decoderInput.saveModel(outputStream);
		
		block1.saveModel(outputStream);
		block2.saveModel(outputStream);
		block3.saveModel(outputStream);

	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		
		decoderInput.loadModel(inputStream);
		
		block1.loadModel(inputStream);
		block2.loadModel(inputStream);
		block3.loadModel(inputStream);
		
	}

}
