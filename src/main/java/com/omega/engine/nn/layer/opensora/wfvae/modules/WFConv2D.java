package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * WFConv2D
 * video to image
 * @author Administrator
 */
public class WFConv2D extends Layer {

    public ConvolutionLayer conv;

    private int depth;
    private int kernelSize;
    private int padding;
    private int stride;
    
    public int oDepth;
    
    private Tensor inputT;
    
    public WFConv2D(int channel, int oChannel, int depth, int height, int width, int kernelSize, int padding, int stride, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.kernelSize = kernelSize;
        this.padding = padding;
        this.stride = stride;
        initLayers();
        this.oDepth = depth;
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
        this.oChannel = conv.oChannel;
    }

    public void initLayers() {
        conv = new ConvolutionLayer(channel, oChannel, width, height, kernelSize, kernelSize, padding, stride, true, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(inputT == null || inputT.number != this.number * depth) {
        	inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
        	this.output = Tensor.createGPUTensor(output, number, conv.oChannel * depth, conv.oHeight, conv.oWidth, true);
        }
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
    	Tensor_OP().permute(input, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
    	conv.forward(inputT);
    	Tensor_OP().permute(conv.getOutput(), output, new int[] {number, depth, conv.oChannel, conv.oHeight, conv.oWidth}, new int[] {number, conv.oChannel, depth, conv.oHeight, conv.oWidth}, new int[]{0, 2, 1, 3, 4});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, conv.getOutput(), new int[] {number, conv.oChannel, depth, conv.oHeight, conv.oWidth}, new int[] {number, depth, conv.oChannel, conv.oHeight, conv.oWidth}, new int[]{0, 2, 1, 3, 4});
        conv.back(conv.getOutput(), inputT);
        Tensor_OP().permute(inputT, input, new int[] {number, depth, channel, height, width}, new int[] {number, channel, depth, height, width}, new int[]{0, 2, 1, 3, 4});
        this.diff = input;
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

