package com.omega.engine.nn.network;

import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.clip.bert.Bert;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import java.io.IOException;
import java.io.RandomAccessFile;

public class ClipText extends Network {
    public int embedDim;
    public int vocabSize;
    public int hiddenSize;
    public int headNum = 8;
    public int maxPositionEmbeddingsSize;
    public int typeVocabSize;
    public int intermediateSize;
    public int numHiddenLayers = 1;
    public Bert bert;
    public Tensor textProjection;
    private InputLayer inputLayer;
    private Tensor tokenIds;
    private Tensor output;

    public ClipText(LossType lossType, UpdaterType updater, int headNum, int time, int vocabSize, int hiddenSize, int embedDim, int maxPositionEmbeddingsSize, int typeVocabSize, int intermediateSize, int numHiddenLayers) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.embedDim = embedDim;
        this.maxPositionEmbeddingsSize = maxPositionEmbeddingsSize;
        this.typeVocabSize = typeVocabSize;
        this.intermediateSize = intermediateSize;
        this.numHiddenLayers = numHiddenLayers;
        this.updater = updater;
        this.headNum = headNum;
        this.time = time;
        this.hiddenSize = hiddenSize;
        this.inputLayer = new InputLayer(1, 1, vocabSize);
        this.bert = new Bert(headNum, time, hiddenSize, vocabSize, maxPositionEmbeddingsSize, typeVocabSize, intermediateSize, numHiddenLayers, this);
        this.addLayer(inputLayer);
        this.addLayer(bert);
        if (textProjection == null) {
            textProjection = new Tensor(1, 1, hiddenSize, embedDim, true);
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
        return NetworkType.CLIP_TEXT;
    }

    @Override
    public Tensor predict(Tensor input) {
        // TODO Auto-generated method stub
        this.RUN_MODEL = RunModel.TEST;
        return this.forward(input);
    }

    @Override
    public Tensor forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        if (tokenIds == null || tokenIds.number != this.number) {
            tokenIds = Tensor.createGPUTensor(tokenIds, number, input.channel, input.height, input.width, true);
        }
        bert.forward(input, tokenIds, null);
        if (output == null || output.number != this.number) {
            output = Tensor.createGPUTensor(output, number, 1, 1, embedDim, true);
        }
        tensorOP.dot(bert.getOutput(), textProjection, output);
        return this.output;
    }

    public Tensor forward(Tensor input, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        if (tokenIds == null || tokenIds.number != this.number) {
            tokenIds = Tensor.createGPUTensor(tokenIds, number, input.channel, input.height, input.width, true);
        }
        bert.forward(input, tokenIds, mask);
        if (output == null || output.number != this.number) {
            output = Tensor.createGPUTensor(output, number, 1, 1, embedDim, true);
        }
        tensorOP.dot(bert.getOutput(), textProjection, output);
        //		output.showDM();
        //		output.showShape();
        return this.output;
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
    }

    public void back(Tensor cos, Tensor sin, Tensor lossDiff) {
        // TODO Auto-generated method stub
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
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

