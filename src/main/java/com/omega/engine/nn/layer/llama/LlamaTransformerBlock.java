package com.omega.engine.nn.layer.llama;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class LlamaTransformerBlock extends Layer{
	
	private int multiple_of = 64;
	
	private int time;
	
	private int headNum = 8;
	
	private int nKVHeads = 8;
	
	private int embedDim = 0;
	
	private boolean bias = false;
	
	private boolean dropout = false;
	
	private boolean flashAttention = false;
	
	private LlamaAttentionLayer attn;
	
	private RMSLayer norm1;
	
	/**
	 * mlp
	 */
	private LlamaMLPLayer mlp;
	private RMSLayer norm2;
	
	private Tensor tmp1;
	
	private Tensor tmp2;
	
	public LlamaTransformerBlock(int headNum,int time,int embedDim,int multiple_of,boolean bias,boolean dropout) {
		this.headNum = headNum;
		this.nKVHeads = headNum;
		this.multiple_of = multiple_of;
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public LlamaTransformerBlock(int headNum,int time,int embedDim,int multiple_of,boolean bias,boolean dropout,boolean flashAttention,Network network) {
		this.flashAttention = flashAttention;
		this.headNum = headNum;
		this.nKVHeads = headNum;
		this.multiple_of = multiple_of;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public LlamaTransformerBlock(int headNum,int nKVHeads,int time,int embedDim,int multiple_of,boolean bias,boolean dropout,boolean flashAttention,Network network) {
		this.flashAttention = flashAttention;
		this.headNum = headNum;
		this.nKVHeads = nKVHeads;
		this.multiple_of = multiple_of;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {

		this.setNorm1(new RMSLayer(this));
		
		if(flashAttention) {
			this.setAttn(new LlamaFlashAttentionLayer(embedDim, headNum, time, bias, dropout, network));
		}else {
			this.setAttn(new LlamaCausalSelfAttention2Layer(embedDim, headNum, nKVHeads, time, bias, dropout, network));
		}

		this.setNorm2(new RMSLayer(getAttn()));
		
		this.setMlp(new LlamaMLPLayer(embedDim, embedDim, multiple_of, bias, network));
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		this.time = this.network.time;
		
		if(this.tmp1 == null || this.tmp1.number != this.number) {
//			if(this.tmp1 == null) {
//				System.out.println(number+":"+embedDim);
//			}
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

	}
	
	public void output(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		
		getNorm1().forward(input);

		getAttn().forward(cos, sin, getNorm1().getOutput());

		Tensor_OP().add(getAttn().getOutput(), input, tmp1);

		getNorm2().forward(tmp1);
		
		getMlp().forward(getNorm2().getOutput());
		
		Tensor_OP().add(getMlp().getOutput(), tmp1, tmp2);
		
		this.output = tmp2;
//		System.err.println("---------------------------------");
//		this.output.showDM();
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
		
//		delta.showDM();
		
		getMlp().back(delta);

		getNorm2().back(getMlp().diff);
		
		Tensor_OP().add(getNorm2().diff, delta, getNorm2().diff);
		
//		norm2.diff.showDM();
//		long start26 = System.nanoTime();
		getAttn().back(cos, sin, getNorm2().diff);
//		System.out.println("atten-back:"+(System.nanoTime() - start26) / 1e6+"ms.");
		
		getNorm1().back(getAttn().diff);
		
		Tensor_OP().add(getNorm1().diff, getNorm2().diff, tmp2);
		
		this.diff = tmp2;
//		System.err.println("diff:");
//		diff.showDM();
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
		
	}
	
	public void forward(Tensor cos,Tensor sin,Tensor input) {
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
		this.output(cos, sin);
		
	}
	
	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

	}
	
	public void back(Tensor cos,Tensor sin,Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(cos, sin);
		
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
	
	public void putParamters() {
		getNorm1().putParamters();
		getAttn().putParamters();
		getNorm2().putParamters();
		getMlp().putParamters();
	}
	
	public void putParamterGrads() {
		getNorm1().putParamterGrads();
		getAttn().putParamterGrads();
		getNorm2().putParamterGrads();
		getMlp().putParamterGrads();
	}

	public LlamaAttentionLayer getAttn() {
		return attn;
	}

	public void setAttn(LlamaAttentionLayer attn) {
		this.attn = attn;
	}

	public RMSLayer getNorm1() {
		return norm1;
	}

	public void setNorm1(RMSLayer norm1) {
		this.norm1 = norm1;
	}

	public LlamaMLPLayer getMlp() {
		return mlp;
	}

	public void setMlp(LlamaMLPLayer mlp) {
		this.mlp = mlp;
	}

	public RMSLayer getNorm2() {
		return norm2;
	}

	public void setNorm2(RMSLayer norm2) {
		this.norm2 = norm2;
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		getNorm1().accGrad(scale);
		getAttn().accGrad(scale);
		getNorm2().accGrad(scale);
		getMlp().accGrad(scale);
	}
	
}
