package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

/**
 * GNLayer3D
 *
 * @author Administrator
 */
public class GNLayer3D extends Layer {

    private int groupNum;
    
    public int depth;
    public int oDepth;
    
    public GNLayer norm;
    
    public GNLayer3D(int channel,int depth, int height, int width,int groupNum, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.groupNum = groupNum;
        initLayers();
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
    }
    
    public GNLayer3D(int channel,int depth, int height, int width,int groupNum, Layer preLayer, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.groupNum = groupNum;
        initLayers();
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
    }

    public void initLayers() {
    	//int groupNum, int channel, int height, int width, BNType bnType, Layer preLayer
    	norm = new GNLayer(groupNum, channel, depth * height, width, BNType.conv_bn, network);
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        this.number = input.number;
    }
    
    @Override
    public void initBack() {

    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }
    
    public static void main(String[] args) {
    	int batchSize = 2;
    	int channel = 3;
    	int numFrames = 2;
    	int imageSize = 8;
    	Tensor input = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, true);
    	
        Transformer tf = new Transformer();
        tf.CUDNN = true;
        float[] data = RandomUtils.order(input.dataLength, 0.1f, 0.1f);
        Tensor input2 = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, data, true);
        GNLayer3D norm = new GNLayer3D(channel, numFrames, imageSize, imageSize, 32, tf);
        for (int i = 0; i < 10; i++) {
        	norm.forward(input2);
        	norm.getOutput().showShape();
        	norm.getOutput().showDM();
//            mal.back(delta);
//            mal.diff.showDM();
        }
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
        this.diff = norm.diff.view(input.shape());
    }
    
    public void diff(Tensor diff) {
        // TODO Auto-generated method stub
    	delta.view(number, channel, depth * height, width);
    	input.view(number, channel, depth * height, width);
    	norm.back(delta, diff);
    	delta.viewOrg();
    	input.viewOrg();
        this.diff = norm.diff.view(input.shape());
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
    
    public void back(Tensor delta,Tensor diff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(diff);
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

