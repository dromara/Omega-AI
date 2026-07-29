package com.omega.engine.nn.layer.dit.flux;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * SimpleHeadMLPLayer
 *
 * @author Administrator
 */
public class SimpleHeadMLPLayer extends Layer {
	
    private int embedDim = 0;
    private int nChannel = 1;
    private int zDim = 0;
    private boolean bias = false;

    public RMSLayer norm;
    public FullyLayer linear1;
    private SiLULayer active;
    public FullyLayer linear2;
    
    public SimpleHeadMLPLayer(int embedDim, int nChannel, int zDim, boolean bias, Network network) {
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
    	
        this.norm = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    	
        this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);
        this.linear1.weight.setData(RandomUtils.xavierUniform(embedDim * nChannel, embedDim, nChannel, 1.0f));
        if(this.linear1.bias != null) {
        	this.linear1.bias.clearGPU();
        }
        this.active = new SiLULayer(linear1);
        this.linear2 = new FullyLayer(nChannel, zDim, bias, network);
        this.linear2.weight.setData(RandomUtils.xavierUniform(nChannel * zDim, nChannel, zDim, 1.0f));
        if(this.linear2.bias != null) {
        	this.linear2.bias.clearGPU();
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
    	norm.forward(input);
        linear1.forward(norm.getOutput());
        active.forward(linear1.getOutput());
        linear2.forward(active.getOutput());
        this.output = linear2.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	this.linear2.back(delta);
        active.back(this.linear2.diff);
        linear1.back(active.diff);
        norm.back(linear1.diff);
        this.diff = this.norm.diff;
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
    	norm.update();
        linear1.update();
        linear2.update();
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
        norm.saveModel(outputStream);
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        norm.accGrad(scale);
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }
}

