package com.omega.engine.nn.layer.unet;

import com.omega.common.tensor.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.DoubleConvLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.pooling.PoolingType;

/**
 * conv + bn + activefunction * 2
 *
 * @author Administrator
 */
public class UNetDownLayer extends Layer {
    private ActiveType activeType;
    private PoolingLayer pooling;
    private DoubleConvLayer conv;

    public UNetDownLayer(int channel, int oChannel, int height, int width, ActiveType activeType, Network network) {
        this.network = network;
        this.activeType = activeType;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public void initLayers() {
        this.pooling = new PoolingLayer(channel, width, height, 2, 2, 2, PoolingType.MAX_POOLING, network);
        this.conv = new DoubleConvLayer(channel, oChannel, pooling.oHeight, pooling.oWidth, activeType, network);
        this.oHeight = this.conv.oHeight;
        this.oWidth = this.conv.oWidth;
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
        this.pooling.forward(input);
        this.conv.forward(this.pooling.getOutput());
        this.output = this.conv.output;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
        this.conv.back(delta);
        this.pooling.back(this.conv.diff);
        this.diff = this.pooling.diff;
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
        conv.update();
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
        conv.accGrad(scale);
    }
}

