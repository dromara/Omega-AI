package com.omega.engine.nn.layer.clip.bert;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * BertEncoder Layer
 *
 * @author Administrator
 */
public class BertEncoderLayer extends Layer {
    public List<BertLayer> layers;
    private int time;
    private int headNum;
    private int hiddenSize;
    private int intermediateSize;
    private int numHiddenLayers;

    public BertEncoderLayer(int headNum, int time, int hiddenSize, int intermediateSize, int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
        this.intermediateSize = intermediateSize;
        this.headNum = headNum;
        this.time = time;
        this.hiddenSize = hiddenSize;
        this.channel = 1;
        this.height = 1;
        this.width = hiddenSize;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public BertEncoderLayer(int headNum, int time, int hiddenSize, int intermediateSize, int numHiddenLayers, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.numHiddenLayers = numHiddenLayers;
        this.intermediateSize = intermediateSize;
        this.headNum = headNum;
        this.time = time;
        this.hiddenSize = hiddenSize;
        this.channel = 1;
        this.height = 1;
        this.width = hiddenSize;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        layers = new ArrayList<BertLayer>();
        for (int i = 0; i < numHiddenLayers; i++) {
            BertLayer b = new BertLayer(headNum, time, hiddenSize, intermediateSize, network);
            layers.add(b);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).forward(x);
            x = layers.get(i).getOutput();
        }
        this.output = x;
    }

    public void output(Tensor mask) {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).forward(x, mask);
            x = layers.get(i).getOutput();
            //			System.out.println("i:"+i);
            //			x.showDMByOffset(0, x.width);
        }
        this.output = x;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output(mask);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.bert_encoder;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

