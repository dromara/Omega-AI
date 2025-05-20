package com.omega.engine.nn.layer;

import com.omega.engine.nn.layer.gpu.UpSample3DKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * 3d上采用层
 * @author Administrator
 */
public class UPSample3DLayer extends Layer {
    private int scale = 2;
    public int depth;
    public int oDepth;
    private UpSample3DKernel kernel;

    public UPSample3DLayer(int channel,int depth, int height, int width, int scale) {
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.depth = depth;
        this.scale = scale;
        this.oDepth = depth * scale;
        this.oHeight = this.height * scale;
        this.oWidth = this.width * scale;
    }

    public UPSample3DLayer(int channel,int depth, int height, int width, int scale, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.scale = scale;
        this.oDepth = this.depth * scale;
        this.oHeight = this.height * scale;
        this.oWidth = this.width * scale;
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.output.number != number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
        if (kernel == null) {
            kernel = new UpSample3DKernel(cuda());
        }
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.output.number != number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
        if (kernel == null) {
            kernel = new UpSample3DKernel(cuda());
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.diff.number != number) {
            this.diff = new Tensor(number, channel * depth, height, width, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.forward(input, output, channel, depth, height, width, scale);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        kernel.backward(delta, diff, channel, depth, height, width, scale);
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

