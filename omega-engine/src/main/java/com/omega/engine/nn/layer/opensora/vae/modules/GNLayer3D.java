package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;

/**
 * GNLayer3D
 *
 * @author Administrator
 */
public class GNLayer3D extends Layer {

    private int groupNum;
    
    public int depth;
    public int oDepth;
    
    private GNLayer norm;
    
    public GNLayer3D(int channel,int depth, int height, int width,int groupNum, Layer preLayer, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.groupNum = groupNum;
        initLayers(preLayer);
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
    }

    public void initLayers(Layer preLayer) {
    	//int groupNum, int channel, int height, int width, BNType bnType, Layer preLayer
    	norm = new GNLayer(groupNum, channel, depth * height, width, BNType.conv_bn, preLayer);
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
    	input.view(number, channel, depth * height, width);
    	norm.forward(input);
    	input.viewOrg();
    	this.output = norm.getOutput().view(number, channel * depth, height, width);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	delta.view(number, channel, depth * height, width);
    	input.view(number, channel, depth * height, width);
    	norm.back(delta);
    	delta.viewOrg();
    	input.viewOrg();
        this.diff = norm.diff;
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
    	norm.update();
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
    	norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream);
    }
}

