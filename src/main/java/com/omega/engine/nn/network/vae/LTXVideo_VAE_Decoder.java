package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.videovae.LTXVideoDecoder3d;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * LTXVideo_VAE
 *
 * @author Administrator
 */
public class LTXVideo_VAE_Decoder extends Network {
	
    public float beta = 0.25f;
    public float decay = 0.999f;
    public int num_res_blocks;
    public int patch_size;
    public int patch_size_t;
    public int num_frames;
    public int height;
    public int width;
    public int latendDim;
    public LTXVideoDecoder3d decoder;
    public Tensor vqLoss;
    public int[] block_out_channels;
    public int[]decoder_block_out_channels;
    public boolean[] decoder_spatio_temporal_scaling;
    public int[] layers_per_block;
    public boolean[] spatio_temporal_scaling;
    private InputLayer inputLayer;
    
    public LTXVideo_VAE_Decoder(LossType lossType, UpdaterType updater, int num_frames, int height, int width, int patch_size_t, int patch_size, int[] block_out_channels, int[] layers_per_block, boolean[] spatio_temporal_scaling) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.patch_size_t = patch_size_t;
        this.patch_size = patch_size;
        this.num_frames = num_frames;
        this.height = height;
        this.width = width;
        this.block_out_channels = block_out_channels;
        this.layers_per_block = layers_per_block;
        this.spatio_temporal_scaling = spatio_temporal_scaling;
        this.updater = updater;
        this.latendDim = block_out_channels[0];
        initLayers();
    }
    
    public void initLayers() {
    	initDecoder();
    	this.inputLayer = new InputLayer(3 * num_frames, height, width);
        this.decoder = new LTXVideoDecoder3d(latendDim, 3, num_frames, height, width, patch_size, patch_size_t, decoder_block_out_channels, layers_per_block, decoder_spatio_temporal_scaling, false, this);
        this.addLayer(inputLayer);
        this.addLayer(decoder);
    }
    
    public void initDecoder() {
    	this.decoder_block_out_channels = new int[block_out_channels.length];
    	this.decoder_spatio_temporal_scaling = new boolean[spatio_temporal_scaling.length];
    	for(int i = 0;i<decoder_block_out_channels.length;i++) {
    		decoder_block_out_channels[i] = block_out_channels[block_out_channels.length - 1 - i];
    	}
    	for(int i = 0;i<decoder_spatio_temporal_scaling.length;i++) {
    		decoder_spatio_temporal_scaling[i] = spatio_temporal_scaling[spatio_temporal_scaling.length - 1 - i];
    	}
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

    public Tensor decode(Tensor latent) {
        this.setInputData(latent);
        decoder.forward(latent);
        return decoder.getOutput();
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
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
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

