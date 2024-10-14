package com.omega.engine.nn.layer.clip;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * CausalSelfAttentionLayer
 * @author Administrator
 *
 */
public class CLIPVisionEmbeddingLayer extends Layer{
	
	private int embedDim = 0;
	
	private Tensor classEmbedding;
	
	private Tensor classEmbeddingEx;
	
	private ConvolutionLayer patchEmbedding;
	
	private int numPatches;
	
	private int numPositions;
	
	private EmbeddingIDLayer positionEmbedding;
	
	private Tensor positionIDS;
	
	private Tensor patchEmbedsT;
	
	private Tensor embeddings;
	
	private BaseKernel kernel;
	
	public CLIPVisionEmbeddingLayer(int channel,int height,int width,int embedDim,int imageSize,int patchSize,boolean bias,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.channel = channel;
		this.embedDim = embedDim;
		this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
		this.numPositions = this.numPatches + 1;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		initLayers(channel, imageSize, imageSize, patchSize, bias);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = network.number;
		if(patchEmbedsT == null || patchEmbedsT.number != this.number) {
			int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
			patchEmbedsT = Tensor.createGPUTensor(patchEmbedsT, this.number, pChannel, 1, embedDim, true);
			embeddings = Tensor.createGPUTensor(embeddings, this.number, pChannel + 1, 1, embedDim, true);
			classEmbeddingEx = Tensor.createGPUTensor(classEmbeddingEx, this.number, 1, 1, embedDim, true);
		}
		
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(patchEmbedsT == null || patchEmbedsT.number != this.number) {
			int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
			patchEmbedsT = Tensor.createGPUTensor(patchEmbedsT, this.number, pChannel, 1, embedDim, true);
			embeddings = Tensor.createGPUTensor(embeddings, this.number, pChannel + 1, 1, embedDim, true);
			classEmbeddingEx = Tensor.createGPUTensor(classEmbeddingEx, this.number, 1, 1, embedDim, true);
		}
	}
	
	public void initLayers(int inChannel,int height,int width,int patchSize,boolean bias) {
		
		this.patchEmbedding = new ConvolutionLayer(inChannel, embedDim, height, width, patchSize, patchSize, 0, patchSize, false, network);
//		float[] wdata = RandomUtils.order(patchEmbedding.weight.dataLength, 0.001f, 0.001f);
//		patchEmbedding.weight = new Tensor(embedDim, inChannel, patchSize, patchSize, wdata, true);
		
		this.positionEmbedding = new EmbeddingIDLayer(numPositions, embedDim, false, network);
//		float[] ewdata = RandomUtils.order(positionEmbedding.weight.dataLength, 0.01f, 0.01f);
//		positionEmbedding.weight = new Tensor(1, 1, numPositions, embedDim, ewdata, true);
		
		if(classEmbedding == null) {
//			float[] data = RandomUtils.order(embedDim, 0.1f, 0.1f);
			classEmbedding = new Tensor(1, 1, 1, embedDim, true);
		}
		
		if(positionIDS == null) {
			float[] data = RandomUtils.order(numPositions, 1.0f, 0.0f);
			positionIDS = new Tensor(numPositions, 1, 1, 1, data, true);
		}
		
		if(kernel == null) {
			kernel = new BaseKernel();
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
		getPatchEmbedding().weight.showShape();
		getPatchEmbedding().weight.showDMByNumber(0);
		getPatchEmbedding().weight.showDMByNumber(767);
		
		getPatchEmbedding().forward(this.input);
		
		getPatchEmbedding().getOutput().showDM();
		
		TensorOP.permute(getPatchEmbedding().getOutput().view(this.number, getPatchEmbedding().getOutput().channel, 1, getPatchEmbedding().getOutput().height * getPatchEmbedding().getOutput().width), patchEmbedsT, new int[] {0, 3, 2, 1});
		
//		patchEmbedsT.showDM();
		
		/**
		 * embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
		 */
		TensorOP.expand(getClassEmbedding(), classEmbeddingEx, getClassEmbedding().getDataLength());
//		classEmbeddingEx.showDM();
		int offset = 0;
		int part_input_size = classEmbeddingEx.getOnceSize() / 1;
		for(int n = 0;n<this.number;n++) {
			kernel.copy_gpu(classEmbeddingEx, this.embeddings, part_input_size, n * classEmbeddingEx.getOnceSize(), 1, offset + n * embeddings.getOnceSize(), 1);
		}
		offset += part_input_size;
		part_input_size = patchEmbedsT.getOnceSize() / 1;
		for(int n = 0;n<this.number;n++) {
			kernel.copy_gpu(patchEmbedsT, this.embeddings, part_input_size, n * patchEmbedsT.getOnceSize(), 1, offset + n * embeddings.getOnceSize(), 1);
		}
//		embeddings.showShape();
//		embeddings.showDM();
		
		/**
		 * embeddings = embeddings + self.position_embedding(self.position_ids)
		 */
		getPositionEmbedding().forward(positionIDS);
		
//		positionEmbedding.getOutput().showDM();
		
		TensorOP.addAxis(this.embeddings, getPositionEmbedding().getOutput(), this.embeddings, getPositionEmbedding().getOutput().getOnceSize());
		
		this.output = this.embeddings;
		
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

	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.clip_vision_embedding;
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
		
		int batchSize = 2;
		int embedDim = 6;
		int imgSize = 224;
		int patchSize = 32;
		
		
		Transformer tf = new Transformer();
		
		tf.number = batchSize;
		
		CLIPVisionEmbeddingLayer layer = new CLIPVisionEmbeddingLayer(3, imgSize, imgSize, embedDim, imgSize, patchSize, false, tf);
		
		float[] data = RandomUtils.order(batchSize * 3 * imgSize * imgSize, 1f, 0f);
		
		Tensor input = new Tensor(batchSize , 3, imgSize, imgSize, data, true);
		
		layer.forward(input);
		
		layer.getOutput().showShape();
		layer.getOutput().showDM();
		
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {

	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

	public Tensor getClassEmbedding() {
		return classEmbedding;
	}

	public ConvolutionLayer getPatchEmbedding() {
		return patchEmbedding;
	}

	public EmbeddingIDLayer getPositionEmbedding() {
		return positionEmbedding;
	}

}
