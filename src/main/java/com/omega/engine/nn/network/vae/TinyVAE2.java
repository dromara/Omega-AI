package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.tae.TAEDecoder;
import com.omega.engine.nn.layer.tae.TAEEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterType;

/**
 * Llama-2
 * @author Administrator
 *
 */
public class TinyVAE2 extends Network {
	
	public float kl_weight = 0.1f;
	
	public int latendDim = 4;
	
	public int imageSize;
	
	public int[] encoderNumBlocks;
	
	public int[] encoderBlockOutChannels;
	
	public int[] decoderNumBlocks;
	
	public int[] decoderBlockOutChannels;
	
	private InputLayer inputLayer;
	
	private TAEEncoder encoder;
	
	private TAEDecoder decoder;
	
	private Tensor z;
	
	private Tensor eps;
	
	private Tensor mu;
	
	private Tensor logvar;
	
	private Tensor dmu;
	
	private Tensor dlogvar;
	
	private VAEKernel vaeKernel;
	
	public Tensor klLoss;
	
	public Tensor encoderDelta;
	
	public TinyVAE2(LossType lossType,UpdaterType updater,int latendDim,int imageSize,int[] encoderNumBlocks,int[] encoderBlockOutChannels,int[] decoderNumBlocks,int[] decoderBlockOutChannels) {
		this.lossFunction = LossFactory.create(lossType, this);
		this.latendDim = latendDim;
		this.imageSize = imageSize;
		this.encoderNumBlocks = encoderNumBlocks;
		this.updater = updater;
		this.encoderBlockOutChannels = encoderBlockOutChannels;
		this.decoderNumBlocks = decoderNumBlocks;
		this.decoderBlockOutChannels = decoderBlockOutChannels;
	
		initLayers();
	}
	
	public void initLayers() {
		this.inputLayer = new InputLayer(3, imageSize, imageSize);
		this.encoder = new TAEEncoder(3, latendDim * 2, imageSize, imageSize, encoderNumBlocks, encoderBlockOutChannels, this);
		this.decoder = new TAEDecoder(latendDim, 3, encoder.oHeight, encoder.oWidth, 2, decoderNumBlocks, decoderBlockOutChannels, this);
		this.addLayer(inputLayer);
		this.addLayer(encoder);
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
		
//		System.out.println("init params.");
//		
//		this.fullyLayer.weight = new Tensor(1, 1, this.fullyLayer.oWidth, this.fullyLayer.width, RandomUtils.gaussianRandom(this.fullyLayer.weight.dataLength, 0.0f, 0.02f), true);
		
		System.out.println("the network is ready.");
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.VAE;
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
//		System.out.println("en_time:"+en_time+",de_time:"+de_time);
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		inputLayer.forward();
		
		encoder.forward(input);
		
		encoder.getOutput().showShape();
		
		reparameterize(encoder.getOutput());
		
		this.z.showShape();
		
		decoder.forward(this.z);
		
		return this.getOutput();
	}
	
	public Tensor encode(Tensor input) {
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		inputLayer.forward();
		
		encoder.forward(input);
		
		reparameterize(encoder.getOutput());
		
		return z;
	}
	
	public Tensor decode(Tensor latent) {
		
		decoder.forward(latent);
		
		return decoder.getOutput();
	}
	
	public void reparameterize(Tensor encode) {
		if(this.z == null || this.z.number != encode.number) {
			this.z = Tensor.createGPUTensor(this.z, encode.number, this.latendDim, encode.height, encode.width, true);
			this.eps = Tensor.createGPUTensor(this.eps, encode.number, this.latendDim, encode.height, encode.width, true);
			this.mu = Tensor.createGPUTensor(this.mu, encode.number, this.latendDim, encode.height, encode.width, true);
			this.logvar = Tensor.createGPUTensor(this.logvar, encode.number, this.latendDim, encode.height, encode.width, true);
		}
		
		GPUOP.getInstance().cudaRandn(this.eps);
		
		vaeKernel.concat_channel_backward(encode, mu, logvar, encode.number, this.latendDim, this.latendDim, encode.height, encode.width);

		vaeKernel.forward(mu, logvar, eps, z);

	}
	
	public void reparameterize_back(Tensor delta) {

		vaeKernel.backward(delta, eps, logvar, dmu, dlogvar);
		
		vaeKernel.concat_channel_forward(dmu, dlogvar, encoderDelta, dmu.number, this.latendDim, this.latendDim, dmu.height, dmu.width);

	}
	
	public void initBack() {
		if(this.dlogvar == null || this.dlogvar.number != logvar.number) {
			this.dlogvar = Tensor.createGPUTensor(this.dlogvar, logvar.number, this.latendDim, logvar.height, logvar.width, true);
			this.dmu = Tensor.createGPUTensor(this.dmu, mu.number, this.latendDim, mu.height, mu.width, true);
			this.encoderDelta = Tensor.createGPUTensor(this.encoderDelta, mu.number, this.latendDim * 2, mu.height, mu.width, true);
		}else {
			dmu.clearGPU();
			dlogvar.clearGPU();
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
		
		// dmu , dlogvar
		vaeKernel.kl_back(mu, logvar, kl_weight, dmu, dlogvar);
		
		this.decoder.back(lossDiff);
		
		reparameterize_back(decoder.diff);
		
		this.encoder.back(encoderDelta);
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label);
	}
	
	public float totalLoss(Tensor output, Tensor label) {

		if(klLoss == null || klLoss.number != mu.number) {
			this.klLoss = Tensor.createTensor(this.klLoss, mu.number, mu.channel, mu.height, mu.width, true);
		}
		
		Tensor decoerLoss = this.lossFunction.loss(output, label);
		
		vaeKernel.kl(mu, logvar, kl_weight, klLoss);
		
		return (MatrixOperation.sum(decoerLoss.syncHost()) + MatrixOperation.sum(klLoss.syncHost())) / input.number;
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
