package com.omega.engine.nn.layer.t5;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.active.GeluType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * T5 feed-forward with gating: wo(GELU(wi_0(x)) * wi_1(x))
 * @author Administrator
 */
public class DenseRelu extends Layer {
    private int embed_size = 0;
    private int ff_size = 1;
    private boolean bias = false;

    public FullyLayer linear1;
    public FullyLayer linear2;
    public FullyLayer linear3;
    private GeluLayer active;

    public DenseRelu(int embed_size, int ff_size, boolean bias) {
        this.embed_size = embed_size;
        this.ff_size = ff_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.initLayers();
    }

    public DenseRelu(int embed_size, int ff_size, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embed_size = embed_size;
        this.ff_size = ff_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(embed_size, ff_size, bias, network);
        this.linear2 = new FullyLayer(embed_size, ff_size, bias, network);
        this.linear3 = new FullyLayer(ff_size, embed_size, bias, network);
        this.active = new GeluLayer(linear1, GeluType.TANH);
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
//    	input.showDM("input");
        linear1.forward(input);
//        linear1.getOutput().showDM("act0:");
        active.forward(linear1.getOutput(), linear1.getOutput());
//        linear1.getOutput().showDM("act1:");
        linear2.forward(input);
        Tensor_OP().mul(active.getOutput(), linear2.getOutput(), linear2.getOutput());
        linear3.forward(linear2.getOutput(), input);
        this.output = linear3.getOutput();
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
        return LayerType.mlp;
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
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
        linear3.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
        linear3.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
        linear3.accGrad(scale);
    }
}

