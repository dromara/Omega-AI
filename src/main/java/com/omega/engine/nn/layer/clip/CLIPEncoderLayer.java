package com.omega.engine.nn.layer.clip;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class CLIPEncoderLayer extends Layer{
	
	private int time;
	
	private int headNum = 8;
	
	private int embedDim = 0;
	
	private int intermediateSize = 3072;
	
	private boolean bias = false;
	
	private boolean mask;
	
	private CLIPAttentionLayer attn;
	
	private LNLayer norm1;
	
	/**
	 * mlp
	 */
	private CLIPMLPLayer mlp;
	
	private LNLayer norm2;
	
	private Tensor tmp1;
	
	private Tensor tmp2;
	
	public CLIPEncoderLayer(int headNum,int time,int embedDim,int intermediateSize,boolean bias,boolean dropout,boolean mask) {
		this.headNum = headNum;
		this.time = time;
		this.intermediateSize = intermediateSize;
		this.embedDim = embedDim;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.mask = mask;
		this.initLayers();
	}
	
	public CLIPEncoderLayer(int headNum,int time,int embedDim,int intermediateSize,boolean bias,boolean dropout,boolean mask,Network network) {
		this.headNum = headNum;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.intermediateSize = intermediateSize;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.mask = mask;
		this.initLayers();
	}
	
	public void initLayers() {

		norm1 = new LNLayer(this);
		
		attn = new CLIPAttentionLayer(embedDim, headNum, time, bias, false, mask, network);

		norm2 = new LNLayer(attn);
		
		mlp = new CLIPMLPLayer(embedDim, intermediateSize, bias, network);
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.time = this.network.time;
		if(this.tmp1 == null || this.tmp1.number != this.number) {
			this.tmp1 = Tensor.createGPUTensor(this.tmp1, number, 1, 1, embedDim, true);
			this.tmp2 = Tensor.createGPUTensor(this.tmp2, number, 1, 1, embedDim, true);
		}
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		this.time = this.network.time;
		if(this.tmp1 == null || this.tmp1.number != this.number) {
			this.tmp1 = Tensor.createGPUTensor(this.tmp1, number, 1, 1, embedDim, true);
			this.tmp2 = Tensor.createGPUTensor(this.tmp2, number, 1, 1, embedDim, true);
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

		getNorm1().forward(input);

		getAttn().forward(getNorm1().getOutput());

		Tensor_OP().add(getAttn().getOutput(), input, tmp1);
		
		getNorm2().forward(tmp1);

		getMlp().forward(getNorm2().getOutput());
		
		Tensor_OP().add(getMlp().getOutput(), tmp1, tmp2);
		
		this.output = tmp2;
		
	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

	}
	
	public void diff(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void back() {
		// TODO Auto-generated method stub

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
		this.init(input);
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
		getNorm1().update();
		getAttn().update();
		getNorm2().update();
		getMlp().update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.transformer_decoder;
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
		getNorm1().saveModel(outputStream);
		getAttn().saveModel(outputStream);
		getNorm2().saveModel(outputStream);
		getMlp().saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		getNorm1().loadModel(inputStream);
		getAttn().loadModel(inputStream);
		getNorm2().loadModel(inputStream);
		getMlp().loadModel(inputStream);
	}

	public CLIPAttentionLayer getAttn() {
		return attn;
	}

	public void setAttn(CLIPAttentionLayer attn) {
		this.attn = attn;
	}

	public LNLayer getNorm1() {
		return norm1;
	}

	public void setNorm1(LNLayer norm1) {
		this.norm1 = norm1;
	}

	public CLIPMLPLayer getMlp() {
		return mlp;
	}

	public void setMlp(CLIPMLPLayer mlp) {
		this.mlp = mlp;
	}

	public LNLayer getNorm2() {
		return norm2;
	}

	public void setNorm2(LNLayer norm2) {
		this.norm2 = norm2;
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub

	}
	
}
