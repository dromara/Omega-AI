package com.omega.engine.nn.layer;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.gpu.ShotcutKernel;
import com.omega.engine.nn.network.Network;

public class ShortcutLayer extends Layer {
    private int c1;
    private int c2;
    private int h1;
    private int h2;
    private int w1;
    private int w2;
    private ShotcutKernel kernel;
    private ShotcutKernel back_kernel;

    public ShortcutLayer(int c1, int h1, int w1, int c2, int h2, int w2, Network network) {
        this.network = network;
        this.c1 = c1;
        this.c2 = c2;
        this.h1 = h1;
        this.h2 = h2;
        this.w1 = w1;
        this.w2 = w2;
        initParam();
        initKernel();
    }

    public void initKernel() {
        kernel = new ShotcutKernel(channel, width, height, oChannel, oWidth, oHeight, cuda());
        back_kernel = new ShotcutKernel(oChannel, oWidth, oHeight, channel, width, height, cuda());
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
        this.width = w2;
        this.height = h2;
        this.channel = c2;
        this.oWidth = w1;
        this.oHeight = h1;
        this.oChannel = c1;
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.shortcut(input, output);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        back_kernel.shortcut(delta, diff);
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
    }

    public void forward(Tensor inpnut, Tensor output) {
        this.output = output;
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(inpnut);
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

    public void back(Tensor delta, Tensor diff) {
        // TODO Auto-generated method stub
        this.diff = diff;
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
        return LayerType.shortcut;
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

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

