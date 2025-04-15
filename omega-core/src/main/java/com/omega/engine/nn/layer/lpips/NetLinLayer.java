package com.omega.engine.nn.layer.lpips;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * NetLinLayer
 *
 * @author Administrator
 */
public class NetLinLayer extends Layer {
    public ConvolutionLayer conv;
    private boolean dropout = false;
    private DropoutLayer drop;

    public NetLinLayer(int channel, int oChannel, int height, int width, boolean dropout, Network network) {
        this.network = network;
        this.dropout = dropout;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public void initLayers() {
        if (dropout) {
            drop = new DropoutLayer(0.5f, network);
        }
        conv = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, false, true, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.relu;
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }

    @Override
    public void initBack() {
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        Tensor x = this.input;
        if (dropout) {
            drop.forward(x);
            x = drop.getOutput();
        }
        conv.forward(x);
        this.output = conv.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println(index);
        Tensor diffOut = delta;
        conv.back(diffOut);
        diffOut = conv.diff;
        if (dropout) {
            drop.back(diffOut);
            diffOut = drop.diff;
        }
        this.diff = diffOut;
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
        return LayerType.block;
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
    }
}

