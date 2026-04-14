package com.omega.engine.nn.network;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.t5.T5Stack;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * T5Encoder
 *
 * @author Administrator
 */
public class T5Encoder extends Network {
	private int voc_size = 250112;
	private int num_layers = 24;
	private int headNum = 32;
	private int time = 120;
    private int embed_size = 2048;
    private int d_ff = 5120;
    private boolean bias = false;
    private InputLayer inputLayer;
    public T5Stack stack;

    public T5Encoder(LossType lossType, UpdaterType updater, int voc_size, int num_layers, int headNum, int time, int embed_size, int d_ff, boolean bias) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.bias = bias;
        this.voc_size = voc_size;
        this.num_layers = num_layers;
        this.headNum = headNum;
        this.time = time;
        this.embed_size = embed_size;
        this.d_ff = d_ff;
        this.inputLayer = new InputLayer(1, 1, voc_size);
        this.stack = new T5Stack(this.voc_size, this.headNum, this.time, this.embed_size, this.d_ff, this.num_layers, this.bias, this);
        this.addLayer(inputLayer);
        this.addLayer(stack);
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
        return NetworkType.ASR;
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

    public Tensor forward(Tensor input, Tensor mask) {
        /**
         * 设置输入数据
         *
         */
        this.setInputData(input);
        inputLayer.forward();
        stack.forward(input, mask);
        return this.getOutput();
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        /**
         * 设置误差 将误差值输入到最后一层
         *
         */
        this.setLossDiff(lossDiff);
        this.stack.back(lossDiff);
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
                // SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
                // softmaxLayer.setCurrentLabel(label);
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
        // PrintUtils.printImage(t.data);
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

    public Tensor lossDiff(Tensor output, Tensor label, int igonre, int count) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre, count);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        stack.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	stack.loadModel(inputStream);
    }

    @Override
    public void putParamters() {
        // transformer.putParamters();
        // getFullyLayer().putParamters();
    }

    @Override
    public void putParamterGrads() {
        // transformer.putParamterGrads();
        // getFullyLayer().putParamterGrads();
    }

}
