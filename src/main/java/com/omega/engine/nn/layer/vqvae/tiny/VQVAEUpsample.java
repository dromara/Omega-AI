package com.omega.engine.nn.layer.vqvae.tiny;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.UPSampleLayer;
import com.omega.engine.nn.network.Network;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * VQVAEUpsample
 *
 * @author Administrator
 */
public class VQVAEUpsample extends Layer {
    private UPSampleLayer up;
    private ConvolutionLayer conv;

    public VQVAEUpsample(int channel, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
    }

    public void initLayers() {
        up = new UPSampleLayer(channel, height, width, 2, network);
        conv = new ConvolutionLayer(channel, oChannel, up.oWidth, up.oHeight, 3, 3, 1, 1, true, this.network);
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
        up.forward(this.input);
        conv.forward(up.getOutput());
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
        conv.back(delta);
        up.back(conv.diff);
        this.diff = up.diff;
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

