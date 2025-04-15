package com.omega.engine.nn.layer.unet;

import com.omega.common.config.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.network.Network;

/**
 * conv + bn + activefunction * 2
 *
 * @author Administrator
 */
public class UNetUPLayer extends Layer {
    private boolean bilinear = true;
    private int kHeight = 3;
    private int kWidth = 3;
    private int padding = 1;
    private int stride = 2;
    private ActiveType activeType;
    private Layer up;
    private DoubleConvLayer conv;
    private RouteLayer cat;
    private Layer x2;

    public UNetUPLayer(int channel, int oChannel, int height, int width, boolean bilinear, Layer x2, ActiveType activeType, Network network) {
        this.network = network;
        this.bilinear = bilinear;
        this.x2 = x2;
        this.activeType = activeType;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.kHeight = 3;
        this.kWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
        this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
        initLayers();
    }

    public void initLayers() {
        Layer[] layers = new Layer[2];
        if (bilinear) {
            this.up = new UPSampleLayer(channel, 2, 2, 1, network);
            this.conv = new DoubleConvLayer(channel, oChannel, channel / 2, up.oHeight, up.oWidth, activeType, network);
        } else {
            this.up = new ConvolutionTransposeLayer(channel, channel / 2, width, height, 2, 2, 0, 2, 1, 0);
            this.conv = new DoubleConvLayer(channel, oChannel, up.oHeight, up.oWidth, activeType, network);
        }
        layers[0] = x2;
        layers[1] = this.up;
        this.cat = new RouteLayer(layers);
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
        this.up.forward(input);
        this.cat.forward();
        this.conv.forward(this.cat.getOutput());
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
        this.cat.back(this.conv.diff);
        this.up.back();
        this.diff = this.up.diff;
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
        up.update();
        conv.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_up;
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
        up.accGrad(scale);
        conv.accGrad(scale);
    }
}

