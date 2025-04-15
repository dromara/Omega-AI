package com.omega.engine.nn.layer;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.gpu.UpSampleKernel;
import com.omega.engine.nn.network.Network;

/**
 * 上采用层
 *
 * @author Administrator
 */
public class UPSampleLayer extends Layer {
    private int stride = 2;
    private UpSampleKernel kernel;

    public UPSampleLayer(int channel, int height, int width, int stride) {
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.stride = stride;
        this.oHeight = this.height * stride;
        this.oWidth = this.width * stride;
        if (this.stride < 0) {
            this.stride = -stride;
            oHeight = this.height / stride;
            oWidth = this.width / stride;
        }
    }

    public UPSampleLayer(int channel, int height, int width, int stride, Network network) {
        this.network = network;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.stride = stride;
        this.oHeight = this.height * stride;
        this.oWidth = this.width * stride;
        if (this.stride < 0) {
            this.stride = -stride;
            oHeight = this.height / stride;
            oWidth = this.width / stride;
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.output.number != number) {
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
        if (kernel == null) {
            kernel = new UpSampleKernel(this.stride, 1.0f, cuda());
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.output.number != number) {
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
        if (kernel == null) {
            kernel = new UpSampleKernel(this.stride, 1.0f, cuda());
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.diff.number != number) {
            this.diff = new Tensor(number, channel, height, width, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.forward(input, output);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        baseKernel().fill_gpu(diff, 0);
        kernel.backward(delta, diff);
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
        return LayerType.upsample;
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

