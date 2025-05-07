package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.cudnn.PoolingCudnnKernel;
import com.omega.engine.nn.layer.gpu.PoolingBaseKernel;
import com.omega.engine.nn.layer.gpu.PoolingKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.pooling.PoolingType;

/**
 * PoolingLayer
 *
 * @author Administrator
 */
public class PoolingLayer extends Layer {
    public PoolingType poolingType;
    public int pWidth = 0;
    public int pHeight = 0;
    public int stride = 1;
    public int padding = 0;
    private PoolingBaseKernel kernel;

    public PoolingLayer(int channel, int width, int height, int pWidth, int pHeight, int stride, PoolingType poolingType) {
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.pWidth = pWidth;
        this.pHeight = pHeight;
        this.stride = stride;
        this.poolingType = poolingType;
        initParam();
    }

    public PoolingLayer(int channel, int width, int height, int pWidth, int pHeight, int stride, PoolingType poolingType, Network network) {
        this.network = network;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.pWidth = pWidth;
        this.pHeight = pHeight;
        this.stride = stride;
        this.poolingType = poolingType;
        initParam();
    }

    public PoolingLayer(int channel, int width, int height, int pWidth, int pHeight, int stride, int padding, PoolingType poolingType) {
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.pWidth = pWidth;
        this.pHeight = pHeight;
        this.stride = stride;
        this.padding = padding;
        this.poolingType = poolingType;
        initParam();
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.output.number != number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
        }
        if (kernel == null) {
            if (this.network.CUDNN) {
                kernel = new PoolingCudnnKernel(poolingType, channel, height, width, oHeight, oWidth, pWidth, pHeight, stride, padding, cuda());
                //				kernel = new PoolingKernel(poolingType, channel, height, width, pHeight, pWidth, stride, padding);
            } else {
                kernel = new PoolingKernel(poolingType, channel, height, width, pHeight, pWidth, stride, padding, cuda());
            }
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (diff == null || this.diff.number != number) {
            this.diff = new Tensor(number, channel, height, width, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        this.oChannel = this.channel;
        this.oWidth = (this.width + padding - pWidth) / this.stride + 1;
        this.oHeight = (this.height + padding - pHeight) / this.stride + 1;
        //		System.out.println("=========>:"+oHeight+":"+height);
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.forward(input, output);
        //		System.out.print("pooling-output:");
        //		System.out.println(output.dataLength);
        //		output.showDM();
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        kernel.backward(input, output, delta, diff);
        //		diff.showDM();
        //		System.out.print("pooling-delta:");
        //		delta.showDM();
        //		System.out.print("pooling-diff:");
        //		diff.showDM();
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
    public void update() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.pooling;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
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
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
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

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

