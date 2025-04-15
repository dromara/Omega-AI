package com.omega.engine.nn.network;

import com.omega.common.tensor.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.model.RNNCellType;
import com.omega.engine.updater.UpdaterType;

/**
 * Recurrent Neural Networks
 *
 * @author Administrator
 */
public class Seq2Seq extends Network {
    public int en_time = 1;
    public int de_time = 1;
    public int en_len;
    public int de_len;
    private RNNCellType cellType;
    /**
     * encoder
     */
    private InputLayer inputLayer;
    private EmbeddingLayer en_emLayer;
    private BaseRNNLayer en_rnnLayer;
    /**
     * decoder
     */
    private EmbeddingLayer de_emLayer;
    private BaseRNNLayer de_rnnLayer;
    private FullyLayer fullyLayer;
    private Tensor en_delta;

    public Seq2Seq(RNNCellType cellType, LossType lossType, UpdaterType updater, int en_time, int de_time, int en_em, int en_hidden, int en_len, int de_em, int de_hidden, int de_len) {
        this.cellType = cellType;
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.en_time = en_time;
        this.de_time = de_time;
        this.en_len = en_len;
        this.de_len = de_len;
        this.inputLayer = new InputLayer(1, 1, en_len);
        this.en_emLayer = new EmbeddingLayer(en_len, en_em, this);
        this.de_emLayer = new EmbeddingLayer(de_len, de_em, this);
        this.fullyLayer = new FullyLayer(de_hidden, de_len, true, this);
        switch (this.cellType) {
            case RNN:
                this.en_rnnLayer = new RNNBlockLayer(en_time, en_em, en_hidden, 1, false, false, 0.0f);
                this.de_rnnLayer = new RNNBlockLayer(de_time, de_em, de_hidden, 1, false, false, 0.0f);
                break;
            case LSTM:
                this.en_rnnLayer = new RNNBlockLayer(en_time, en_em, en_hidden, 2, false, false, 0.0f);
                this.de_rnnLayer = new RNNBlockLayer(de_time, de_em, de_hidden, 2, false, false, 0.0f);
                break;
            case GRU:
                this.en_rnnLayer = new RNNBlockLayer(en_time, en_em, en_hidden, 3, false, false, 0.0f);
                this.de_rnnLayer = new RNNBlockLayer(de_time, de_em, de_hidden, 3, false, false, 0.0f);
                break;
        }
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
        return NetworkType.SEQ2SEQ;
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

    public Tensor forward(Tensor en_input, Tensor de_input) {
        //		System.out.println("en_time:"+en_time+",de_time:"+de_time);
        /**
         * 设置输入数据
         *
         */
        this.setInputData(en_input);
        /**
         * 编码器
         *
         */
        this.inputLayer.forward();
        this.en_emLayer.forward();
        this.en_rnnLayer.forward(this.en_time, this.en_emLayer.output.number);
        //		this.en_rnnLayer.getOutput().showDMByNumber((en_time - 1) * en_rnnLayer.getOutput().number / en_time);
        //		this.en_rnnLayer.getHy().showDMByNumber(0);
        /**
         * 解码器
         *
         */
        this.de_emLayer.forward(de_input);
        this.de_rnnLayer.forward(this.de_emLayer.getOutput(), this.en_rnnLayer.getHy(), this.en_rnnLayer.getCy(), this.de_time);
        this.fullyLayer.forward(this.de_rnnLayer.getOutput());
        return this.getOutput();
    }

    public void initEnRNNLayerDelta(Tensor delta) {
        if (this.en_delta == null || this.en_delta.number != this.en_rnnLayer.getOutput().number) {
            this.en_delta = Tensor.createTensor(this.en_delta, this.en_rnnLayer.getOutput().number, this.en_rnnLayer.getOutput().channel, this.en_rnnLayer.getOutput().height, this.en_rnnLayer.getOutput().width, true);
        }
        this.en_delta.clearGPU();
        //		en_delta.showShape();
        //		delta.showShape();
        /**
         * rnn mode
         *
         */
        //		if(this.en_rnnLayer.rnnMode == 0 || this.en_rnnLayer.rnnMode == 1){
        baseKernel.copy_gpu(delta, this.en_delta, delta.getDataLength(), 0, 1, (en_time - 1) * delta.getDataLength(), 1);
        //		}
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        /**
         * 设置误差 将误差值输入到最后一层
         *
         */
        this.setLossDiff(lossDiff);
        this.fullyLayer.back();
        this.de_rnnLayer.back();
        this.de_emLayer.back();
        initEnRNNLayerDelta(this.de_rnnLayer.getDhx());
        this.en_rnnLayer.back(this.en_delta, this.en_rnnLayer.getHy(), this.en_rnnLayer.getCy(), null, this.de_rnnLayer.getDcx());
        this.en_emLayer.back(this.en_rnnLayer.diff);
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

    public Tensor[] encoder(Tensor input) {
        Tensor[] outputs = new Tensor[3];
        /**
         * 设置输入数据
         *
         */
        this.setInputData(input);
        /**
         * 编码器
         *
         */
        this.inputLayer.forward();
        this.en_emLayer.forward();
        this.en_rnnLayer.forward(this.en_time, this.en_emLayer.getOutput().number);
        outputs[0] = en_rnnLayer.getOutput();
        outputs[1] = en_rnnLayer.getHy();
        outputs[2] = en_rnnLayer.getCy();
        return outputs;
    }

    public Tensor decoder(Tensor hx, Tensor cx, Tensor start) {
        /**
         * 解码器
         *
         */
        this.de_emLayer.forward(start);
        //		h.showDM();
        this.de_rnnLayer.forward(this.de_emLayer.getOutput(), hx, cx, 1);
        baseKernel.copy_gpu(this.de_rnnLayer.getHy(), hx, hx.getDataLength(), 0, 1, 0, 1);
        baseKernel.copy_gpu(this.de_rnnLayer.getCy(), cx, cx.getDataLength(), 0, 1, 0, 1);
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
