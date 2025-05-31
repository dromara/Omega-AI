package com.omega.engine.nn.network;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.asr.ASRTransformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Llama-2
 *
 * @author Administrator
 */
public class ASR extends Network {
    public ASRTransformer transformer;
    public FullyLayer fullyLayer;
    private int wavDim;
    private int wavTime;
    private int vocSize;
    private int vocTime;
    private int embedDim = 0;
    private int nChannel = 1;
    private int headNum = 8;
    private int n_layers = 4;
    private boolean bias = true;
    private InputLayer inputLayer;

    public ASR(LossType lossType, UpdaterType updater, int wavDim, int wavTime, int vocSize, int vocTime, int embedDim, int nChannel, int headNum, int n_layers, boolean bias, boolean dropout) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.bias = bias;
        this.wavDim = wavDim;
        this.wavTime = wavTime;
        this.vocSize = vocSize;
        this.vocTime = vocTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.inputLayer = new InputLayer(1, 1, vocSize);
        this.setTransformer(new ASRTransformer(this.wavDim, this.wavTime, this.vocSize, this.vocTime, this.embedDim, this.headNum, this.nChannel, this.n_layers, this.bias, this));
        this.setFullyLayer(new FullyLayer(embedDim, vocSize, bias, this));
        this.addLayer(inputLayer);
        this.addLayer(getTransformer());
        this.addLayer(getFullyLayer());
    }

    public ASR(LossType lossType, UpdaterType updater, int wavDim, int wavTime, int vocSize, int vocTime, int embedDim, int nChannel, int headNum, int n_layers, boolean bias, boolean dropout, int rankId) {
        super(rankId);
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.bias = bias;
        this.wavDim = wavDim;
        this.wavTime = wavTime;
        this.vocSize = vocSize;
        this.vocTime = vocTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.inputLayer = new InputLayer(1, 1, vocSize);
        this.setTransformer(new ASRTransformer(this.wavDim, this.wavTime, this.vocSize, this.vocTime, this.embedDim, this.headNum, this.nChannel, this.n_layers, this.bias, this));
        this.setFullyLayer(new FullyLayer(embedDim, vocSize, bias, this));
        this.addLayer(inputLayer);
        this.addLayer(getTransformer());
        this.addLayer(getFullyLayer());
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
        // System.out.println("init params.");
        //
        // this.fullyLayer.weight = new Tensor(1, 1, this.fullyLayer.oWidth,
        // this.fullyLayer.width,
        // RandomUtils.gaussianRandom(this.fullyLayer.weight.dataLength, 0.0f, 0.02f),
        // true);
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

    public Tensor forward(Tensor wavInput, Tensor wavLen, Tensor labelInput, Tensor labelLen) {
        /**
         * 设置输入数据
         *
         */
        this.setInputData(wavInput);
        inputLayer.forward();
        getTransformer().forward(wavInput, wavLen, labelInput, labelLen);
        getFullyLayer().forward(getTransformer().getOutput());
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
        this.getFullyLayer().back(lossDiff);
        getTransformer().back(this.getFullyLayer().diff);
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
        transformer.saveModel(outputStream);
        getFullyLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        transformer.loadModel(inputStream);
        getFullyLayer().loadModel(inputStream);
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

    public FullyLayer getFullyLayer() {
        return fullyLayer;
    }

    public void setFullyLayer(FullyLayer fullyLayer) {
        this.fullyLayer = fullyLayer;
    }

    public ASRTransformer getTransformer() {
        return transformer;
    }

    public void setTransformer(ASRTransformer transformer) {
        this.transformer = transformer;
    }
}
