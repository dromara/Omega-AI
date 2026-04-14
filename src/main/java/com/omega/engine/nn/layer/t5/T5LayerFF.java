package com.omega.engine.nn.layer.t5;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * T5LayerFF
 * @author Administrator
 */
public class T5LayerFF extends Layer {
    private int embed_size = 0;
    private int ff_size = 1;
    private boolean bias = false;

    public LNLayer norm;
    public DenseRelu dr;

    public T5LayerFF(int embed_size, int ff_size, boolean bias) {
        this.embed_size = embed_size;
        this.ff_size = ff_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.initLayers();
    }

    public T5LayerFF(int embed_size, int ff_size, boolean bias, Network network) {
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
        this.norm = new LNLayer(1, 1, embed_size, true, false, BNType.fully_bn, network);
        this.dr = new DenseRelu(embed_size, ff_size, bias, network);
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
    	norm.forward_t5(input);
    	dr.forward(norm.getOutput());
    	Tensor_OP().add(input, dr.getOutput(), dr.getOutput());
        this.output = dr.getOutput();
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
        norm.saveModel(outputStream);
        dr.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream);
    	dr.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	norm.accGrad(scale);
    	dr.accGrad(scale);
    }
}

