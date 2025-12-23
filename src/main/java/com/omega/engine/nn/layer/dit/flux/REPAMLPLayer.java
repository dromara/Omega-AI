package com.omega.engine.nn.layer.dit.flux;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * REPAMLPLayer
 *
 * @author Administrator
 */
public class REPAMLPLayer extends Layer {
	
    private int embedDim = 0;
    private int nChannel = 1;
    private int zDim = 0;
    private boolean bias = false;

    public FullyLayer linear1;
    private SiLULayer active1;
    public FullyLayer linear2;
    private SiLULayer active2;
    public FullyLayer linear3;
    
    public REPAMLPLayer(int embedDim, int nChannel, int zDim, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.zDim = zDim;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = zDim;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);
        this.linear1.weight.setData(RandomUtils.xavierUniform(embedDim * nChannel, embedDim, nChannel, 1.0f));
        if(this.linear1.bias != null) {
        	this.linear1.bias.clearGPU();
        }
        this.active1 = new SiLULayer(linear1);
        this.linear2 = new FullyLayer(nChannel, nChannel, bias, network);
        this.linear2.weight.setData(RandomUtils.xavierUniform(nChannel * nChannel, nChannel, nChannel, 1.0f));
        if(this.linear2.bias != null) {
        	this.linear2.bias.clearGPU();
        }
        this.active2 = new SiLULayer(linear2);
        this.linear3 = new FullyLayer(nChannel, zDim, bias, network);
        this.linear3.weight.setData(RandomUtils.xavierUniform(nChannel * zDim, nChannel, zDim, 1.0f));
        if(this.linear3.bias != null) {
        	this.linear3.bias.clearGPU();
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        linear1.forward(input);
        active1.forward(linear1.getOutput());
        linear2.forward(active1.getOutput());
        active2.forward(linear2.getOutput());
        linear3.forward(active2.getOutput());
        this.output = linear3.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	this.linear3.back(this.delta);
    	active2.back(linear3.diff);
    	this.linear2.back(active2.diff);
        active1.back(this.linear2.diff);
        linear1.back(active1.diff);
        this.diff = this.linear1.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 参数初始化
         */
        this.init();
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
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init();
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
        linear1.update();
        linear2.update();
        linear3.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mlp;
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
        linear3.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
        linear3.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
        linear3.accGrad(scale);
    }
}

