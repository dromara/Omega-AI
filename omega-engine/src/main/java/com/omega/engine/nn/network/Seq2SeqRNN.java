package com.omega.engine.nn.network;

import com.omega.engine.active.ActiveType;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * Recurrent Neural Networks
 *
 * @author Administrator
 */
public class Seq2SeqRNN extends Network {
    public int en_time = 1;
    public int de_time = 1;
    public int en_len;
    public int de_len;
    /**
     * encoder
     */
    private InputLayer inputLayer;
    private EmbeddingLayer en_emLayer;
    private RNNLayer en_rnnLayer;
    /**
     * decoder
     */
    private EmbeddingLayer de_emLayer;
    private RNNLayer de_rnnLayer;
    private FullyLayer fullyLayer;
    private Tensor c;
    private Tensor en_delta;

    public Seq2SeqRNN(LossType lossType, UpdaterType updater, int en_time, int de_time, int en_em, int en_hidden, int en_len, int de_em, int de_hidden, int de_len) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.en_time = en_time;
        this.de_time = de_time;
        this.en_len = en_len;
        this.de_len = de_len;
        this.inputLayer = new InputLayer(1, 1, en_len);
        this.en_emLayer = new EmbeddingLayer(en_len, en_em, this);
        this.en_rnnLayer = new RNNLayer(en_em, en_hidden, en_time, ActiveType.tanh, false, this);
        this.de_emLayer = new EmbeddingLayer(de_len, de_em, this);
        this.de_rnnLayer = new RNNLayer(de_em, de_hidden, de_time, ActiveType.tanh, false, this);
        this.fullyLayer = new FullyLayer(de_hidden, de_len, true, this);
        this.addLayer(inputLayer);
        this.addLayer(en_emLayer);
        this.addLayer(en_rnnLayer);
        this.addLayer(de_emLayer);
        this.addLayer(de_rnnLayer);
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
        return NetworkType.SEQ2SEQ_RNN;
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

    public void initHidden(int number, int hiddenSize) {
        int batch = number / en_time;
        if (this.c == null || this.c.number != batch) {
            this.c = Tensor.createTensor(this.c, batch, 1, 1, hiddenSize, true);
        }
        baseKernel.copy_gpu(en_rnnLayer.getOutput(), this.c, this.c.getDataLength(), (en_time - 1) * this.c.getDataLength(), 1, 0, 1);
    }

    public void initHidden(Tensor h, int number, int hiddenSize) {
        int batch = number / en_time;
        if (this.c == null || this.c.number != batch) {
            this.c = Tensor.createTensor(this.c, batch, 1, 1, hiddenSize, true);
        }
        baseKernel.copy_gpu(h, this.c, this.c.getDataLength(), (en_time - 1) * this.c.getDataLength(), 1, 0, 1);
    }

    public void initEnRNNLayerDelta(Tensor delta) {
        if (this.en_delta == null || this.en_delta.number != this.en_rnnLayer.getOutput().number) {
            this.en_delta = Tensor.createTensor(this.en_delta, this.en_rnnLayer.getOutput().number, this.en_rnnLayer.getOutput().channel, this.en_rnnLayer.getOutput().height, this.en_rnnLayer.getOutput().width, true);
        }
        this.en_delta.clearGPU();
        //		en_delta.showShape();
        //		delta.showShape();
        //		System.out.println(delta.getDataLength());
        baseKernel.copy_gpu(delta, this.en_delta, delta.getDataLength(), 0, 1, (en_time - 1) * delta.getDataLength(), 1);
    }

    public Tensor forward(Tensor en_input, Tensor de_input) {
        /**
         * 设置输入数据

         */
        this.setInputData(en_input);
        /**
         * 编码器

         */
        this.inputLayer.forward();
        this.en_emLayer.forward();
        this.en_rnnLayer.forward(en_time, this.en_emLayer.output.number);
        /**
         * 解码器

         */
        this.de_emLayer.forward(de_input);
        this.initHidden(en_rnnLayer.getOutput().number, en_rnnLayer.getOutput().width);
        this.de_rnnLayer.forward(this.de_emLayer.getOutput(), this.c, de_time);
        this.fullyLayer.forward(this.de_rnnLayer.getOutput());
        return this.getOutput();
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);
        //		lossDiff.showDMByNumber(0);
        this.fullyLayer.back();
        this.de_rnnLayer.back();
        this.de_emLayer.back();
        //		this.c.getGrad().showDM();
        this.initEnRNNLayerDelta(this.c.getGrad());
        //		this.en_delta.showDM();
        this.en_rnnLayer.back(this.en_delta);
        this.en_emLayer.back(en_rnnLayer.diff);
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

    public Tensor encoder(Tensor input) {
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        /**
         * 编码器

         */
        this.inputLayer.forward();
        this.en_emLayer.forward();
        this.en_rnnLayer.forward(en_time, this.en_emLayer.getOutput().number);
        return en_rnnLayer.getOutput();
    }

    public Tensor decoder(Tensor h, Tensor start) {
        /**
         * 解码器

         */
        this.de_emLayer.forward(start);
        //		h.showDM();
        this.de_rnnLayer.forward(this.de_emLayer.getOutput(), h, 1);
        baseKernel.copy_gpu(this.de_rnnLayer.getOutput(), h, h.getDataLength(), 0, 1, 0, 1);
        this.fullyLayer.forward(this.de_rnnLayer.getOutput());
        //		this.fullyLayer.getOutput().showDM();
        return this.fullyLayer.getOutput();
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



