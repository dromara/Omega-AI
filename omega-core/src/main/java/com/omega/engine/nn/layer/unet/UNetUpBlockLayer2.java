package com.omega.engine.nn.layer.unet;

import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * UNetUpBlockLayer2
 *
 * @author Administrator
 */
public class UNetUpBlockLayer2 extends Layer {
    public ConvolutionTransposeLayer upSampleConv;
    public UNetResnetBlockLayer2 resnet1;
    public UNetResnetBlockLayer2 resnet2;
    public UNetResnetBlockLayer2 resnet3;
    public UNetSpatialTransformerLayer st1;
    public UNetSpatialTransformerLayer st2;
    private int tEmbDim;
    private int contextDim = 0;
    private int maxContextLen = 0;
    private int numHeads;
    private int groups = 32;

    public UNetUpBlockLayer2(int channel, int oChannel, int height, int width, int tEmbDim, int numHeads, int groups, int contextDim, int maxContextLen, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.tEmbDim = tEmbDim;
        this.contextDim = contextDim;
        this.maxContextLen = maxContextLen;
        this.numHeads = numHeads;
        this.groups = groups;
        initLayers();
    }

    public void initLayers() {
        upSampleConv = new ConvolutionTransposeLayer(channel, oChannel, width, height, 4, 4, 1, 2, 1, 0, true, network);
        upSampleConv.setUpdater(UpdaterFactory.create(this.network));
        upSampleConv.paramsInit = ParamsInit.silu;
        this.oHeight = upSampleConv.oHeight;
        this.oWidth = upSampleConv.oWidth;
        resnet1 = new UNetResnetBlockLayer2(oChannel, oChannel, oHeight, oWidth, tEmbDim, groups, network);
        st1 = new UNetSpatialTransformerLayer(oChannel, oHeight, oWidth, numHeads, maxContextLen, contextDim, true, false, network);
        resnet2 = new UNetResnetBlockLayer2(oChannel, oChannel, oHeight, oWidth, tEmbDim, groups, network);
        st2 = new UNetSpatialTransformerLayer(oChannel, oHeight, oWidth, numHeads, maxContextLen, contextDim, false, true, network);
        resnet3 = new UNetResnetBlockLayer2(oChannel, oChannel, oHeight, oWidth, tEmbDim, groups, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.getShape()[0];
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
        upSampleConv.forward(input);
        resnet1.forward(upSampleConv.getOutput(), tembd);
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
        resnet3.back(delta, tDiff);
        st2.back(resnet3.diff);
        resnet2.back(st2.diff, tDiff);
        st1.back(resnet2.diff);
        resnet1.back(st1.diff, tDiff);
        upSampleConv.back(resnet1.diff);
        this.diff = upSampleConv.diff;
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
        upSampleConv.update();
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
        upSampleConv.accGrad(scale);
        resnet1.accGrad(scale);
        st1.accGrad(scale);
        resnet2.accGrad(scale);
        st2.accGrad(scale);
        resnet3.accGrad(scale);
    }
}

