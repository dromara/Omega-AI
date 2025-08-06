package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.opensora.wfvae.decoder.WFDecoder;
import com.omega.engine.nn.layer.opensora.wfvae.encoder.WFEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * WFVAE
 *
 * @author Administrator
 */
public class WFVAE extends Network {

    public int num_res_blocks;
    public int en_energy_flow_hidden_size;
    public int de_energy_flow_hidden_size;
    public int connect_res_layer_num = 1;
    public int latendDim = 4;
    public int depth;;
    public int imageSize;
    public WFEncoder encoder;
    public WFDecoder decoder;

    private int base_channels;
    private InputLayer inputLayer;
    
    private Tensor dec_t;
    private Tensor input_t;
    
    private Tensor r_z;
    private Tensor z;
    private Tensor mean;
    private Tensor logvar;
    
    private Tensor rec_loss;
    private Tensor kl_loss;
    
    private float wavelet_weight = 0.1f;
    
    private Tensor wl_loss_l1;
    private Tensor wl_loss_l2;
    
    private Tensor loss_sum;
    
    private Tensor lpipsLossDiff;
    
    private Tensor en_l1_coeffs_delta;
    private Tensor en_l2_coeffs_delta;
    private Tensor de_l1_coeffs_delta;
    private Tensor de_l2_coeffs_delta;
    
    private Tensor dmean;
    private Tensor dlogvar;
    
    private VAEKernel vaeKernel;

    public WFVAE(LossType lossType, UpdaterType updater, int depth, int latendDim, int imageSize, int base_channels, int en_energy_flow_hidden_size, int de_energy_flow_hidden_size, int num_res_blocks, int connect_res_layer_num) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.en_energy_flow_hidden_size = en_energy_flow_hidden_size;
        this.de_energy_flow_hidden_size = de_energy_flow_hidden_size;
        this.connect_res_layer_num = connect_res_layer_num;
        this.imageSize = imageSize;
        this.num_res_blocks = num_res_blocks;
        this.base_channels = base_channels;
        this.depth = depth;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        this.encoder = new WFEncoder(3, depth, imageSize, imageSize, num_res_blocks, base_channels, en_energy_flow_hidden_size, latendDim, this);
        this.decoder = new WFDecoder(3, encoder.oDepth, encoder.oHeight, encoder.oWidth, num_res_blocks, base_channels, de_energy_flow_hidden_size, latendDim, connect_res_layer_num, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
        this.addLayer(decoder);
        vaeKernel = new VAEKernel(cudaManager);
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 0) {
            throw new Exception("layer size must greater than 2.");
        }
        this.layerCount = layerList.size();
        this.setChannel(layerList.get(0).channel);
        this.setHeight(layerList.get(0).height);
        this.setWidth(layerList.get(0).width);
        this.oChannel = this.getLastLayer().oChannel;
        this.oHeight = this.getLastLayer().oHeight;
        this.oWidth = this.getLastLayer().oWidth;
        if (layerList.get(0).getLayerType() != LayerType.input) {
            throw new Exception("first layer must be input layer.");
        }
        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
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
        inputLayer.forward();
        encoder.forward(input);
        posterior(encoder.getOutput());
        decoder.forward(z);  //decoder_diff + (kl_mean_diff cat kl_std_diff)
        return this.getOutput();
    }

    public Tensor encode(Tensor input) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
//        encoder.getOutput().showDM("encoder");
        posterior(encoder.getOutput());
        return z;
    }
    
    public Tensor encode(Tensor input, Tensor r_z) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
//        encoder.getOutput().showDM("encoder");
        sample(encoder.getOutput(), r_z);
        return z;
    }

    public Tensor decode(Tensor latent) {
        this.setInputData(latent);
        decoder.forward(latent);
        return decoder.getOutput();
    }

    public void posterior(Tensor encoder_out) {
    	sample(encoder_out);
    }
    
    public void sample(Tensor en_out) {

    	if(z == null || z.number != en_out.number) {
    		mean = Tensor.createGPUTensor(mean, en_out.number, latendDim * encoder.oDepth, en_out.height, en_out.width, true);
    		logvar = Tensor.createGPUTensor(logvar, mean.shape(), true);
    		r_z = Tensor.createGPUTensor(r_z, mean.shape(), true);
    		z = Tensor.createGPUTensor(z, mean.shape(), true);
    	}
    	
    	tensorOP.getByChannel(en_out, mean, new int[] {number, encoder.oChannel, 1, encoder.oDepth * en_out.height * en_out.width}, 0);
    	tensorOP.getByChannel(en_out, logvar, new int[] {number, encoder.oChannel, 1, encoder.oDepth * en_out.height * en_out.width}, latendDim);
    	
    	RandomUtils.gaussianRandom(r_z);
    	//mean + std * rnd
    	vaeKernel.forward(mean, logvar, r_z, z);
    	
    }
    
    public void sample(Tensor en_out, Tensor r_z) {
    	this.r_z = r_z;
    	if(z == null || z.number != en_out.number) {
    		this.number = en_out.number;
    		mean = Tensor.createGPUTensor(mean, en_out.number, latendDim * encoder.oDepth, en_out.height, en_out.width, true);
    		logvar = Tensor.createGPUTensor(logvar, mean.shape(), true);
    		z = Tensor.createGPUTensor(z, mean.shape(), true);
    	}

    	tensorOP.getByChannel(en_out, mean, new int[] {number, encoder.oChannel, 1, encoder.oDepth * en_out.height * en_out.width}, 0);
    	tensorOP.getByChannel(en_out, logvar, new int[] {number, encoder.oChannel, 1, encoder.oDepth * en_out.height * en_out.width}, latendDim);
    	
    	tensorOP.clamp(logvar, -30.0f, 20.0f, logvar);
    	
    	vaeKernel.forward(mean, logvar, r_z, z);
    	
    }
   
    public void initBack() {

    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);

    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }
    
    public float totalLoss(Tensor output, Tensor label, LPIPS lpips) {
//        mean.showDM("mean");
//        logvar.showDM("logvar");
    	if (rec_loss == null || rec_loss.number != output.number) {
    		this.number = output.number;
            lpips.number = number * depth;
    		dec_t = Tensor.createGPUTensor(dec_t, number * depth, 3, imageSize, imageSize, true);
    		input_t = Tensor.createGPUTensor(input_t, number * depth, 3, imageSize, imageSize, true);
            this.rec_loss = Tensor.createGPUTensor(rec_loss, dec_t.shape(), true);
            kl_loss = Tensor.createGPUTensor(kl_loss, mean.shape(), true);
            if(encoder.getL1_coeffs() != null) {
            	wl_loss_l1 = Tensor.createGPUTensor(wl_loss_l1, encoder.getL1_coeffs().shape(), true);
                wl_loss_l2 = Tensor.createGPUTensor(wl_loss_l2, decoder.getL1_coeffs().shape(), true);
            }
            loss_sum = Tensor.createGPUTensor(loss_sum, 1, 1, 1, 1, true);
        }
    	
    	float loss = 0.0f;
        int bs = output.number;
    	int[] xShape = new int[] {number, 3, depth, imageSize, imageSize};
    	int[] tShape = new int[] {number, depth, 3, imageSize, imageSize};
    	
    	tensorOP.permute(output, dec_t, xShape, tShape, new int[] {0, 2, 1, 3, 4});
    	tensorOP.permute(label, input_t, xShape, tShape, new int[] {0, 2, 1, 3, 4});
    	
//    	dec_t.showDM("dec_t");
    	//l1 = abs(x - p)
    	//重建损失
        vaeKernel.l1_loss(dec_t, input_t, rec_loss);
        //lpips质量损失
        Tensor lpipsOutput = lpips.forward(dec_t, input_t);

//        lpipsOutput.showDM();
        //kl散度损失
        vaeKernel.kl(mean, logvar, 1, kl_loss);
        
        //wf_loss
        if(encoder.getL1_coeffs() != null) {
        	encoder.getL1_coeffs().showShape();
        	decoder.getL1_coeffs().showShape();
        	vaeKernel.l1_loss(encoder.getL1_coeffs(), decoder.getL1_coeffs(), wl_loss_l1);
            vaeKernel.l1_loss(encoder.getL2_coeffs(), decoder.getL2_coeffs(), wl_loss_l2);
        }
        
//        tensorOP.sum(rec_loss, loss_sum, 0);
//        float recLossSum = loss_sum.syncHost()[0];
//        System.out.println("recLossSum:" + recLossSum);
//        loss = loss + recLoss;
        
//        tensorOP.sum(lpipsOutput, loss_sum, 0);
        
        tensorOP.add(rec_loss, lpipsOutput, rec_loss, rec_loss.getOnceSize());
//        rec_loss.showDM("weighted_nll_loss");
        tensorOP.sum(rec_loss, loss_sum, 0);
        float recLoss = loss_sum.syncHost()[0] / rec_loss.number;
        System.out.println("recLoss:" + recLoss);
        loss = loss + recLoss;
        
//        lpipsOutput.showDM("lpipsOutput");
//        float lpipsLoss = loss_sum.syncHost()[0] / rec_loss.number;
//        System.out.println("lpipsLoss:" + lpipsLoss);
//        loss = loss + lpipsLoss;
        
//        kl_loss.showDM("kl_loss");
        tensorOP.sum(kl_loss, loss_sum, 0);
        float klLoss = loss_sum.syncHost()[0] / bs;
        System.out.println("klLoss:" + klLoss);
        loss = loss + klLoss;
        
        if(encoder.getL1_coeffs() != null) {
	        tensorOP.sum(wl_loss_l1, loss_sum, 0);
	        float l1Loss = loss_sum.syncHost()[0] * wavelet_weight / bs;
//	        System.out.println("l1Loss:" + l1Loss);
	        loss = loss + l1Loss;
	        
	        tensorOP.sum(wl_loss_l2, loss_sum, 0);
	        float l2Loss = loss_sum.syncHost()[0] * wavelet_weight / bs;
//	        System.out.println("l2Loss:" + l2Loss);
	        System.out.println("wf_loss:" + (l1Loss + l2Loss));
	        loss = loss + l2Loss;
        }

//        System.err.println("loss:" + loss);
        
        return loss;
    }
    
    public void backward(LPIPS lpips) {
    	int bs = mean.number;
    	Tensor deltaT = rec_loss;
    	Tensor delta = decoder.getOutput();
    	if(en_l1_coeffs_delta == null || en_l1_coeffs_delta.number != bs) {
    		if(encoder.getL1_coeffs() != null) {
	    		en_l1_coeffs_delta = Tensor.createGPUTensor(en_l1_coeffs_delta, encoder.getL1_coeffs().shape(), true);
	    		en_l2_coeffs_delta = Tensor.createGPUTensor(en_l2_coeffs_delta, encoder.getL2_coeffs().shape(), true);
	    		de_l1_coeffs_delta = Tensor.createGPUTensor(de_l1_coeffs_delta, decoder.getL1_coeffs().shape(), true);
	    		de_l2_coeffs_delta = Tensor.createGPUTensor(de_l2_coeffs_delta, decoder.getL2_coeffs().shape(), true);
    		}
    		lpipsLossDiff = new Tensor(dec_t.number, 1, 1, 1, MatrixUtils.val(dec_t.number, 1.0f / dec_t.number * encoder.channel * encoder.height * encoder.width), true);
    		dmean = Tensor.createGPUTensor(dmean, mean.shape(), true);
    		dlogvar = Tensor.createGPUTensor(dlogvar, logvar.shape(), true);
    	}
    	
    	if(encoder.getL1_coeffs() != null) {
    		
    		vaeKernel.l1_loss_allBack(encoder.getL1_coeffs(), decoder.getL1_coeffs(), en_l1_coeffs_delta, de_l1_coeffs_delta, wl_loss_l1.number);
        	vaeKernel.l1_loss_allBack(encoder.getL2_coeffs(), decoder.getL2_coeffs(), en_l2_coeffs_delta, de_l2_coeffs_delta, wl_loss_l2.number);
        	
        	tensorOP.mul(en_l1_coeffs_delta, wavelet_weight, en_l1_coeffs_delta);
        	tensorOP.mul(en_l2_coeffs_delta, wavelet_weight, en_l2_coeffs_delta);
        	tensorOP.mul(de_l1_coeffs_delta, wavelet_weight, de_l1_coeffs_delta);
        	tensorOP.mul(de_l2_coeffs_delta, wavelet_weight, de_l2_coeffs_delta);
    	}
    	
    	//y = x^2 x‘=2x^2-1 = 2x
    	// 0.5f * (-1 + logvar[i] + powf(mu[i], 2) + expf(logvar[i])) * kl_weight;
    	vaeKernel.kl_back(mean, logvar, 1.0f / bs, dmean, dlogvar);
    	
//    	lpipsLossDiff.showDM();
    	
    	lpips.back(lpipsLossDiff);
    	
//    	lpips.lpips.diff.showDM("lpips.diff");
    	
    	vaeKernel.l1_loss_back(dec_t, input_t, deltaT, dec_t.number);
//    	lpips.lpips.diff.showDM("lpips.diff");
//    	deltaT.showDM("deltaT");
    	tensorOP.add(deltaT, lpips.lpips.diff, deltaT);

    	int[] tShape = new int[] {number, 3, depth, imageSize, imageSize};
    	int[] xShape = new int[] {number, depth, 3, imageSize, imageSize};
    	
    	tensorOP.permute(deltaT, delta, xShape, tShape, new int[] {0, 2, 1, 3, 4});
    	
//    	delta.showDM("delta");
    	
    	decoder.back(delta, de_l1_coeffs_delta, de_l2_coeffs_delta);
    	//sample backward
    	vaeKernel.backward(decoder.diff, r_z, logvar, dmean, dlogvar);
    	
//    	dmean.showDM();
    	
    	tensorOP.getByChannel_back(encoder.getOutput(), dmean, new int[] {number, encoder.oChannel, 1, encoder.oDepth * encoder.oHeight * encoder.oWidth}, 0);
    	tensorOP.getByChannel_back(encoder.getOutput(), dlogvar, new int[] {number, encoder.oChannel, 1, encoder.oDepth * encoder.oHeight * encoder.oWidth}, latendDim);
    	
//    	encoder.getOutput().showDM();
    	
    	encoder.back(encoder.getOutput(), en_l1_coeffs_delta, en_l2_coeffs_delta);
    }
    
    public float totalLoss(Tensor output, Tensor label) {

    	return 0.0f;
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

    public Tensor loss(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, igonre);
    }

    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        encoder.saveModel(outputStream);
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        encoder.loadModel(inputStream);
        decoder.loadModel(inputStream);
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

