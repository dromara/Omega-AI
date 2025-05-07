package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * Downsample3D
 *
 * @author Administrator
 */
public class Downsample3D extends Layer {

    private int stride;
    
    public int depth;
    public int oDepth;
    
    private CausalConv3DPlainAR conv3d;
    
    public Downsample3D(int channel,int depth, int height, int width,int stride, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.stride = stride;
        initLayers();
        this.oChannel = conv3d.oChannel;
        this.oDepth = conv3d.oDepth;
        this.oHeight = conv3d.oHeight;
        this.oWidth = conv3d.oWidth;
    }

    public void initLayers() {
    	
    	conv3d = new CausalConv3DPlainAR(channel, channel, depth, width, height, 3, stride, true, network);
    	
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
    	conv3d.forward(input);
    	this.output = conv3d.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	conv3d.back(delta);
        this.diff = conv3d.diff;
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
        conv3d.update();
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
    	conv3d.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv3d.loadModel(inputStream);
    }
}

