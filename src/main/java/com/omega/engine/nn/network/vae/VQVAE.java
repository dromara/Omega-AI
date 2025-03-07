package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.vqvae.VQVAEDecoder;
import com.omega.engine.nn.layer.vqvae.VQVAEEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;

import jcuda.jcublas.cublasOperation;

/**
 * VQVAE
 * @author Administrator
 *
 */
public class VQVAE extends Network {
	
	public float beta = 0.2f;
	
	private int num_vq_embeddings;
	
	public int headNum = 4;
	
	public int latendDim = 4;
	
	public int imageSize;
	
	private int numLayers;
	
	private int[] downChannels;
	
	private boolean[] downSample;
	
	private int[] midChannels;
	
	private InputLayer inputLayer;
	
	public VQVAEEncoder encoder;
	
	public VQVAEDecoder decoder;
	
	public ConvolutionLayer pre_quant_conv;
	
	public EmbeddingIDLayer embedding;
	
	public ConvolutionLayer post_quant_conv;
	
	private Tensor zq;
	
	private Tensor ze;
	
	private Tensor z_flattened;
	
//	private Tensor cd;
	
	private Tensor idx;

	private Tensor dzqT;
	
	private Tensor dze;
	
	private Tensor dzeT;
	
	private VAEKernel vaeKernel;
	
	public Tensor vqLoss;
	
	private Tensor zc;
	
	private Tensor ec;
	
	private Tensor ie;
	
	public VQVAE(LossType lossType,UpdaterType updater,int latendDim,int imageSize,int numLayers,int headNum,int num_vq_embeddings,int[] downChannels,boolean[] downSample,int[] midChannels) {
		this.lossFunction = LossFactory.create(lossType, this);
		this.downChannels = downChannels;
		this.downSample = downSample;
		this.num_vq_embeddings = num_vq_embeddings;
		this.midChannels = midChannels;
		this.latendDim = latendDim;
		this.headNum = headNum;
		this.numLayers = numLayers;
		this.imageSize = imageSize;
		this.updater = updater;
		initLayers();
	}
	
	public void initLayers() {
		this.inputLayer = new InputLayer(3, imageSize, imageSize);
		this.encoder = new VQVAEEncoder(3, latendDim, imageSize, imageSize, numLayers, 32, headNum, downChannels, downSample, midChannels, this);
		
		pre_quant_conv = new ConvolutionLayer(latendDim, latendDim, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
		pre_quant_conv.setUpdater(UpdaterFactory.create(this.updater, this.updaterParams));
		pre_quant_conv.paramsInit = ParamsInit.silu;
		
		embedding = new EmbeddingIDLayer(num_vq_embeddings, latendDim, this);
		
		post_quant_conv = new ConvolutionLayer(latendDim, latendDim, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
		post_quant_conv.setUpdater(UpdaterFactory.create(this.updater, this.updaterParams));
		post_quant_conv.paramsInit = ParamsInit.silu;

		this.decoder = new VQVAEDecoder(latendDim, 3, encoder.oHeight, encoder.oWidth, numLayers, 32, headNum, downChannels, downSample, midChannels, this);
		this.addLayer(inputLayer);
		this.addLayer(encoder);
		this.addLayer(pre_quant_conv);
		this.addLayer(embedding);
		this.addLayer(post_quant_conv);
		this.addLayer(decoder);
		
		vaeKernel = new VAEKernel(cudaManager);
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		if(layerList.size() <= 0) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		this.setChannel(layerList.get(0).channel);
		this.setHeight(layerList.get(0).height);
		this.setWidth(layerList.get(0).width);
		
		this.oChannel = this.getLastLayer().oChannel;
		this.oHeight = this.getLastLayer().oHeight;
		this.oWidth = this.getLastLayer().oWidth;
		
		if(layerList.get(0).getLayerType() != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}

		System.out.println("the network is ready.");
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.VQVAE;
	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOutput();
	}
	
	@Override
	public Tensor forward(Tensor input) {
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
//		input.showDMByNumber(0);
		inputLayer.forward();
//		long start1 = System.nanoTime();
		encoder.forward(input);
//		System.out.println("encoder:"+(System.nanoTime() - start1)/1e6+"ms.");
		pre_quant_conv.forward(encoder.getOutput());
//		long start2 = System.nanoTime();
		quantizer();
//		System.out.println("quantizer:"+(System.nanoTime() - start2)/1e6+"ms.");
		post_quant_conv.forward(this.zq);
//		long start3 = System.nanoTime();
		decoder.forward(post_quant_conv.getOutput());

//		System.out.println("decoder:"+(System.nanoTime() - start3)/1e6+"ms.");
		return this.getOutput();
	}
	
	public Tensor encode(Tensor input) {
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		inputLayer.forward();
		
		encoder.forward(input);
		
		pre_quant_conv.forward(encoder.getOutput());
		
		quantizer();
		
		return zq;
	}
	
	public Tensor decode(Tensor latent) {

		post_quant_conv.forward(latent);

		decoder.forward(post_quant_conv.getOutput());
		
		return decoder.getOutput();
	}
	
	public void quantizer() {

		this.ze = pre_quant_conv.getOutput();

		if(this.z_flattened == null || this.z_flattened.number != ze.number) {
			this.z_flattened = Tensor.createGPUTensor(this.z_flattened, ze.number, ze.height, ze.width, this.latendDim, true);
			this.idx = Tensor.createGPUTensor(this.idx, ze.number * ze.height * ze.width, 1, 1, 1, true);
			this.zq = Tensor.createGPUTensor(this.zq, ze.number, ze.channel, ze.height, ze.width, true);
		}else {
			z_flattened.viewOrg();
		}
	
		tensorOP.permute(ze, z_flattened, new int[] {0, 2, 3, 1});  //B,C,H,W ==> B,H,W,C

		z_flattened = z_flattened.view(ze.number * ze.height * ze.width, 1, 1, this.latendDim);

		cdist();
		vaeKernel.argmin(ie, idx);

		if(embedding.getOutput() != null) {
			embedding.getOutput().viewOrg();
		}

		embedding.forward(idx);

		Tensor emo = embedding.getOutput().view(new int[] {ze.number, ze.height, ze.width, ze.channel});
		
		tensorOP.permute(emo, zq, new int[] {0, 3, 1, 2}); //B*H*W*C ==> B*C*H*W
		
	}
	
	/**
	 * sqrt(sum(pow(x - y))) ==> sum(x^2) + sum(y^2) - 2 * x * y
	 * @return
	 */
	public void cdist() {
		if(zc == null || z_flattened.number != zc.number) {
			zc = Tensor.createGPUTensor(this.zc, z_flattened.number, 1, 1, 1, true);
			ec = Tensor.createGPUTensor(this.ec, num_vq_embeddings, 1, 1, 1, true);
			ie = Tensor.createGPUTensor(this.ie, z_flattened.number, 1, 1, num_vq_embeddings, true);
		}else {
			ie.clearGPU();
			zc.clearGPU();
			ec.clearGPU();
		}
//		long start1 = System.nanoTime();
		tensorOP.sum_pow(z_flattened, zc, 2, 1);
		tensorOP.sum_pow(embedding.weight.view(num_vq_embeddings, 1, 1, latendDim), ec, 2, 1);

		tensorOP.broadcast(zc, ie, 1);

		tensorOP.broadcast_row(ec, ie);

		GPUOP.getInstance().multiplyFloat(z_flattened.number, embedding.weight.number, embedding.weight.width, z_flattened.getGpuData(), embedding.weight.getGpuData(), ie.getGpuData(),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, -2.0f, 1.0f);
		
		embedding.weight.viewOrg();
//		JCuda.cudaDeviceSynchronize();
//		System.err.println("cdist:"+(System.nanoTime() - start1)/1e6);
	}
	
	public Tensor quantizer_back(Tensor delta) {

		dzqT.viewOrg();
		
//		TensorOP.permute(delta, dzqT, new int[] {0, 2, 3, 1}); // B,C,H,W ==> B,H,W,C
		
		// dzq , dze
		vaeKernel.MSE_BACK(embedding.getOutput(), z_flattened, dzqT, dzeT, beta);
		
		tensorOP.permute(dzeT, dze, new int[] {0, 3, 1, 2});  //B,H,W,C ==》 B,C,H,W

		dzqT.view(dzqT.number * dzqT.height * dzqT.width, 1, 1, latendDim);
		embedding.back(dzqT);
		
		tensorOP.add(dze, delta, dze);

		return dze;
	}
	
	public void initBack() {
		if(this.dzqT == null || this.dzqT.number != zq.number) {
			this.dzqT = Tensor.createGPUTensor(this.dzqT, zq.number, zq.height, zq.width, zq.channel, true);
			this.dze = Tensor.createGPUTensor(this.dze, ze.number, ze.channel, ze.height, ze.width, true);
			this.dzeT = Tensor.createGPUTensor(this.dzeT, zq.number, zq.height, zq.width, zq.channel, true);
		}
	}
	
	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub
//		lossDiff.showDMByNumber(0);
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);  //only decoder delta
		
		initBack();

		this.decoder.back(lossDiff);
		
		post_quant_conv.back(decoder.diff); //decoder diff
		
		Tensor encoderDelta = quantizer_back(post_quant_conv.diff);
		
		pre_quant_conv.back(encoderDelta);
		
		this.encoder.back(pre_quant_conv.diff);
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label);
	}
	
	public float totalLoss(Tensor output, Tensor label) {

		if(vqLoss == null) {
			this.vqLoss = Tensor.createTensor(this.vqLoss, 1, 1, 1, 1, true);
		}
		
		Tensor decoerLoss = this.lossFunction.loss(output, label);
		System.out.println(MatrixOperation.sum(decoerLoss.syncHost()));
		embedding.getOutput().viewOrg();
		
		vaeKernel.MSE(embedding.getOutput(), z_flattened, vqLoss, beta);
		
		vqLoss.showDM();

		return (MatrixOperation.sum(decoerLoss.syncHost()) / input.number + MatrixOperation.sum(vqLoss.syncHost()));
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		Tensor t = this.lossFunction.diff(output, label);
		return t;
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, loss);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, diff);
	}
	
	public Tensor loss(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, igonre);
	}

	public Tensor lossDiff(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, igonre);
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {

	}

	@Override
	public void putParamters() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void putParamterGrads() {
		// TODO Auto-generated method stub
		
	}
	
}
