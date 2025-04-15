package com.omega.engine.nn.network;

import com.omega.common.config.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.transformer.TransformerDecoder;
import com.omega.engine.updater.UpdaterType;

/**
 * Recurrent Neural Networks
 *
 * @author Administrator
 */
public class GPT extends Network {
    public int time = 1;
    public int vocab_size;
    public int embedDim;
    public int nChannel;
    public int head_num = 8;
    private InputLayer inputLayer;
    private TransformerDecoder decoder;
    private FullyLayer fullyLayer;

    public GPT(LossType lossType, UpdaterType updater, int vocab_size, int time, int embedDim, int nChannel) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.time = time;
        this.vocab_size = vocab_size;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.inputLayer = new InputLayer(1, 1, vocab_size);
        this.decoder = new TransformerDecoder(vocab_size, time, embedDim, nChannel, true, true, this);
        this.fullyLayer = new FullyLayer(embedDim, vocab_size, true, this);
        this.addLayer(inputLayer);
        this.addLayer(decoder);
        this.addLayer(fullyLayer);
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
        return NetworkType.GPT;
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
        // TODO Auto-generated method stub
        return this.getOutput();
    }

    public Tensor forward(Tensor input, Tensor positions, Tensor mask) {
        //		System.out.println("en_time:"+en_time+",de_time:"+de_time);
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        decoder.forward(input, mask, positions);
        //		decoder.getOutput().showShape();
        fullyLayer.forward(decoder.getOutput());
        return this.getOutput();
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);
        this.fullyLayer.back(lossDiff);
        this.decoder.back(this.fullyLayer.diff);
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
                //			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
                //			softmaxLayer.setCurrentLabel(label);
                break;
            case softmax_cross_entropy:
                SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) this.getLastLayer();
                softmaxWithCrossEntropyLayer.setCurrentLabel(label);
                break;
            default:
                break;
        }
        return this.lossFunction.loss(output, label);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        Tensor t = this.lossFunction.diff(output, label);
        //		PrintUtils.printImage(t.data);
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

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
}

