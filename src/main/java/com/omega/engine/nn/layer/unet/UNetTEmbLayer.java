package com.omega.engine.nn.layer.unet;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * t_emb_layers
 *
 * @author Administrator
 */
public class UNetTEmbLayer extends Layer {
    public FullyLayer linear;
    private SiLULayer act;

    public UNetTEmbLayer(int t_emb_dim, int out_channels, Network network) {
        this.network = network;
        this.channel = 1;
        this.height = 1;
        this.width = t_emb_dim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = width;
        initLayers(out_channels);
    }

    public void initLayers(int oc) {
        act = new SiLULayer(this);
        linear = new FullyLayer(width, oc, true, network);
        this.oWidth = oc;
        //		System.out.println("activeLayer.preLayer:"+activeLayer.preLayer);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
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
        act.forward(input);
        linear.forward(act.getOutput());
        this.output = linear.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		delta.showDM("delta");
        linear.back(delta);
        //		linear.diff.showDM("tl");
        act.back(linear.diff);
        this.diff = act.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        linear.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
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
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        linear.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear.loadModel(inputStream);
    }
}

