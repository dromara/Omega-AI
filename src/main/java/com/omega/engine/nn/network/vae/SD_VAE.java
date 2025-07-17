package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEDecoder;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * TinyVAE
 *
 * @author Administrator
 */
public class SD_VAE extends Network {
	
    public float beta = 0.25f;
    public float decay = 0.999f;
    public int num_res_blocks;
    public int num_vq_embeddings;
    public int latendDim = 4;
    public int imageSize;
    public SDVAEEncoder encoder;
    public SDVAEDecoder decoder;
    public ConvolutionLayer pre_quant_conv;
    public ConvolutionLayer post_quant_conv;
    public Tensor vqLoss;
    private int groups = 32;
    private int headNum = 4;
    public int[] ch_mult;
    private int ch;
    private InputLayer inputLayer;
    
    private Tensor mean;
    private Tensor logvar;
    private Tensor std;
    private Tensor z;
    
    private boolean double_z = true;

    public SD_VAE(LossType lossType, UpdaterType updater,int latendDim, int num_vq_embeddings, int imageSize, int[] ch_mult, int ch, int num_res_blocks, boolean double_z) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.num_vq_embeddings = num_vq_embeddings;
        this.imageSize = imageSize;
        this.ch_mult = ch_mult;
        this.num_res_blocks = num_res_blocks;
        this.ch = ch;
        this.double_z = double_z;
        this.updater = updater;
        initLayers();
    }
    
    public void initLayers() {
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        if(double_z) {
        	this.encoder = new SDVAEEncoder(3, latendDim * 2, imageSize, imageSize, num_res_blocks, groups, headNum, ch_mult, ch, false, this);
        	pre_quant_conv = new ConvolutionLayer(latendDim * 2, latendDim * 2, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
        }else {
        	this.encoder = new SDVAEEncoder(3, latendDim, imageSize, imageSize, num_res_blocks, groups, headNum, ch_mult, ch, false, this);
        	pre_quant_conv = new ConvolutionLayer(latendDim, latendDim, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
        }
        post_quant_conv = new ConvolutionLayer(latendDim, latendDim, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
        this.decoder = new SDVAEDecoder(latendDim, 3, encoder.oHeight, encoder.oWidth, num_res_blocks, groups, headNum, ch_mult, ch, false, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
        this.addLayer(pre_quant_conv);
        this.addLayer(post_quant_conv);
        this.addLayer(decoder);

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
        return NetworkType.SD_VAE;
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
    	return null;
    }

    public Tensor encode(Tensor input) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
        pre_quant_conv.forward(encoder.getOutput());
        sample(pre_quant_conv.getOutput());
        return z;
    }

    public Tensor decode(Tensor latent) {
        this.setInputData(latent);
        post_quant_conv.forward(latent);
        decoder.forward(post_quant_conv.getOutput());
        return decoder.getOutput();
    }
    
    public void sample(Tensor en_out) {
    	
    	if(z == null || z.number != en_out.number) {
    		mean = Tensor.createGPUTensor(mean, en_out.number, latendDim, en_out.height, en_out.width, true);
    		logvar = Tensor.createGPUTensor(logvar, mean.shape(), true);
    		std = Tensor.createGPUTensor(std, mean.shape(), true);
    		z = Tensor.createGPUTensor(z, mean.shape(), true);
    	}
    	
    	RandomUtils.gaussianRandom(z);
    	
    	tensorOP.getByChannel(en_out, mean, 0, latendDim);
    	tensorOP.getByChannel(en_out, logvar, latendDim, latendDim);
    	
    	tensorOP.clamp(logvar, -30, 20, logvar);
    	
    	tensorOP.mul(logvar, 0.5f, std);
    	tensorOP.exp(std, std);
    	
    	tensorOP.mul(z, std, z);
    	tensorOP.add(z, mean, z);
    }
    
    public void initBack() {
       
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
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
        pre_quant_conv.saveModel(outputStream);
        post_quant_conv.saveModel(outputStream);
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        encoder.loadModel(inputStream);
        pre_quant_conv.loadModel(inputStream);
        post_quant_conv.loadModel(inputStream);
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

