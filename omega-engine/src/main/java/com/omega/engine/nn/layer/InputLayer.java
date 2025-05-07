package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;

public class InputLayer extends Layer {
    public InputLayer(int channel, int height, int width) {
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        this.input = this.network.input;
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
        this.output = this.input;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.diff = this.network.getNextLayer(0).diff;
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        this.init();
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
        this.diff();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
        //		System.out.println("input layer not have diff.");
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.input;
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

