package com.omega.engine.nn.network;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.transformer.TransformerNanoDecoder;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Recurrent Neural Networks
 *
 * @author Administrator
 */
public class NanoGPT extends Network {
    public int vocabSize;
    public int embedDim;
    public int headNum = 8;
    public int decoderNum = 1;
    private boolean bias = true;
    private boolean dropout;
    private InputLayer inputLayer;
    private TransformerNanoDecoder decoder;
    private FullyLayer fullyLayer;

    public NanoGPT(LossType lossType, UpdaterType updater, int headNum, int decoderNum, int vocabSize, int time, int embedDim, boolean bias, boolean dropout) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.bias = bias;
        this.dropout = dropout;
        this.decoderNum = decoderNum;
        this.updater = updater;
        this.headNum = headNum;
        this.time = time;
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.inputLayer = new InputLayer(1, 1, vocabSize);
        this.decoder = new TransformerNanoDecoder(this.vocabSize, this.decoderNum, this.headNum, this.time, this.embedDim, this.bias, this.dropout, this);
        this.fullyLayer = new FullyLayer(embedDim, vocabSize, false, this);
        this.addLayer(inputLayer);
        this.addLayer(decoder);
        this.addLayer(fullyLayer);
    }

    public NanoGPT(LossType lossType, UpdaterType updater, int headNum, int decoderNum, int vocabSize, int time, int embedDim, boolean bias, boolean dropout, boolean cudnn) {
        this.CUDNN = cudnn;
        this.lossFunction = LossFactory.create(lossType, this);
        this.bias = bias;
        this.dropout = dropout;
        this.decoderNum = decoderNum;
        this.updater = updater;
        this.headNum = headNum;
        this.time = time;
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.inputLayer = new InputLayer(1, 1, vocabSize);
        this.decoder = new TransformerNanoDecoder(this.vocabSize, this.decoderNum, this.headNum, this.time, this.embedDim, this.bias, this.dropout, this);
        this.fullyLayer = new FullyLayer(embedDim, vocabSize, false, this);
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

    public Tensor forward(Tensor input, Tensor positions) {
        //		System.out.println("en_time:"+en_time+",de_time:"+de_time);
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        decoder.forward(input, positions);
        fullyLayer.forward(decoder.getOutput());
        return this.getOutput();
    }

    //
    //	public Tensor forward(Tensor input,Tensor positions,Tensor mask) {
    ////		System.out.println("en_time:"+en_time+",de_time:"+de_time);
    //		/**
    //		 * 设置输入数据
    //		 */
    //		this.setInputData(input);
    //
    //		inputLayer.forward();
    //
    //		decoder.forward(input, mask, positions);
    //
    //		fullyLayer.forward(decoder.getOutput());
    //
    //		return this.getOutput();
    //	}
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        decoder.saveModel(outputStream);
        System.out.println("decoder save success...");
        fullyLayer.saveModel(outputStream);
        System.out.println("fullyLayer save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        decoder.loadModel(inputStream);
        fullyLayer.loadModel(inputStream);
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

