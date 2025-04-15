package com.omega.engine.nn.layer.unet;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * UNetMidBlockLayer2
 *
 * @author Administrator
 */
public class UNetMidBlockLayer2 extends Layer {
    public UNetResnetBlockLayer2 resnet1;
    public UNetResnetBlockLayer2 resnet2;
    public UNetSpatialTransformerLayer st1;
    public UNetSpatialTransformerLayer st2;
    public UNetResnetBlockLayer2 resnet3;
    private int tEmbDim;
    private int contextDim = 0;
    private int maxContextLen = 0;
    private int numHeads;
    private int groups = 32;
    ;

    public UNetMidBlockLayer2(int channel, int oChannel, int height, int width, int tEmbDim, int numHeads, int groups, int contextDim, int maxContextLen, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        this.tEmbDim = tEmbDim;
        this.contextDim = contextDim;
        this.maxContextLen = maxContextLen;
        this.numHeads = numHeads;
        this.groups = groups;
        initLayers();
    }

    public void initLayers() {
        resnet1 = new UNetResnetBlockLayer2(channel, oChannel, height, width, tEmbDim, groups, network);
        st1 = new UNetSpatialTransformerLayer(oChannel, height, width, numHeads, maxContextLen, contextDim, true, true, network);
        resnet2 = new UNetResnetBlockLayer2(oChannel, oChannel, height, width, tEmbDim, groups, network);
        st2 = new UNetSpatialTransformerLayer(oChannel, height, width, numHeads, maxContextLen, contextDim, true, true, network);
        resnet3 = new UNetResnetBlockLayer2(oChannel, oChannel, height, width, tEmbDim, groups, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
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
    }

    public void output(Tensor tembd, Tensor context) {
        // TODO Auto-generated method stub
        resnet1.forward(input, tembd);
        st1.forward(resnet1.getOutput(), context);
        resnet2.forward(st1.getOutput(), tembd);
        st2.forward(resnet2.getOutput(), context);
        resnet3.forward(st2.getOutput(), tembd);
        this.output = resnet3.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    }

    public void diff(Tensor tDiff) {
        // TODO Auto-generated method stub
        //		delta.showDM("c+d");
        resnet3.back(delta, tDiff);
        st2.back(resnet3.diff);
        resnet2.back(st2.diff, tDiff);
        st1.back(resnet2.diff);
        resnet1.back(st1.diff, tDiff);
        this.diff = resnet1.diff;
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
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor tembd, Tensor context) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(tembd, context);
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

    public void back(Tensor delta, Tensor tDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(tDiff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        resnet1.update();
        st1.update();
        resnet2.update();
        st2.update();
        resnet3.update();
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
        resnet1.accGrad(scale);
        st1.accGrad(scale);
        resnet2.accGrad(scale);
        st2.accGrad(scale);
        resnet3.accGrad(scale);
    }
}

