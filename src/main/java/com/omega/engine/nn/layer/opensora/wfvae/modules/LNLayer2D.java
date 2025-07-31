package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

/**
 * LNLayer2D
 *
 * @author Administrator
 */
public class LNLayer2D extends Layer {

    public LNLayer norm;
    
    private int[] xShape;
    private int[] tShape;
    
    private Tensor inputT;
    
    public LNLayer2D(int channel, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
    }
    
    public void initLayers() {
    	//int groupNum, int channel, int height, int width, BNType bnType, Layer preLayer
    	norm = new LNLayer(height, width, channel, BNType.fully_bn, network);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(inputT == null || inputT.number != this.number) {
        	inputT = Tensor.createGPUTensor(inputT, number, height, width, channel, true);
        	this.output = Tensor.createGPUTensor(output, number, channel, height, width, true);
        	xShape = new int[] {number, channel, height, width};
        	tShape = new int[] {number, height, width, channel};
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if(inputT == null || inputT.number != this.number) {
        	inputT = Tensor.createGPUTensor(inputT, number, height, width, channel, true);
        	this.output = Tensor.createGPUTensor(output, number, channel, height, width, true);
        	xShape = new int[] {number, channel, height, width};
        	tShape = new int[] {number, height, width, channel};
        }
    }
    
    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.number != this.number) {
    		diff = Tensor.createGPUTensor(diff, number, channel, height, width, true);
    	}
    }
    
    public void initBack(Tensor diff) {
    	this.diff = diff;
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
        LNLayer2D norm = new LNLayer2D(channel, imageSize, imageSize, tf);
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
    	Tensor_OP().permute(input, inputT, xShape, tShape, new int[] {0, 2, 3, 1});
    	norm.forward_llmc(inputT);
    	Tensor_OP().permute(norm.getOutput(), output, tShape, xShape, new int[] {0, 3, 1, 2});
//    	output.showDM("norm");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, norm.getOutput(), xShape, tShape, new int[] {0, 2, 3, 1});
    	norm.back(norm.getOutput());
    	Tensor_OP().permute(norm.diff, diff, tShape, xShape, new int[] {0, 3, 1, 2});
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
        initBack(diff);
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

