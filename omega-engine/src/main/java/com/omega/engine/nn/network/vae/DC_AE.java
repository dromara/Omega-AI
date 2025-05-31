package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dc_ae.DCAEDecoder;
import com.omega.engine.nn.layer.dc_ae.DCAEEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * DC_AE
 *
 * @author Administrator
 */
public class DC_AE extends Network {

    public int imageSize;
    public DCAEEncoder encoder;
    public DCAEDecoder decoder;
   
    private int groups = 32;
    private int initChannel;
    private int num_blocks;
    private int num_stages;
    private int latent_channels;

    private InputLayer inputLayer;
    

    //	private Tensor avg_probs;
    //	private Tensor avg_probs_log;
    public DC_AE(LossType lossType, UpdaterType updater, int initChannel, int latent_channels, int imageSize, int num_blocks, int spatial_compression, int groups) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.initChannel = initChannel;
        this.latent_channels = latent_channels;
        this.imageSize = imageSize;
        this.num_blocks = num_blocks;
        this.num_stages = (int) (Math.log(spatial_compression) / Math.log(2));
        System.err.println(num_stages);
        this.groups = groups;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        this.encoder = new DCAEEncoder(3, initChannel, imageSize, imageSize, groups, num_blocks, num_stages, latent_channels, this);
        this.decoder = new DCAEDecoder(3, initChannel, imageSize, imageSize, groups, num_blocks, num_stages, latent_channels, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
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
        return NetworkType.DCAE;
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
        encoder.getOutput().showShape();
        decoder.forward(encoder.getOutput());
        return this.getOutput();
    }

    public Tensor encode(Tensor input) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
        return encoder.getOutput();
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
        //		lossDiff.showDMByNumber(0);
        /**
         * 设置误差
         * 将误差值输入到最后一层
         */
        this.setLossDiff(lossDiff);  //only decoder delta
        
        initBack();

        this.decoder.back(lossDiff);
       
        this.encoder.back(decoder.diff);
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

