package com.omega.engine.nn.layer.dit.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTSimpleHeadLayer
 *
 * @author Administrator
 */
public class DiTSimpleHeadLayer extends Layer {
    private int iChannel = 0;
    private int oChannel = 1;
    private boolean bias = false;

    public FullyLayer linear1;
    private GeluLayer active;
    public FullyLayer linear2;

    public DiTSimpleHeadLayer(int iChannel, int oChannel, boolean bias) {
        this.iChannel = iChannel;
        this.oChannel = oChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = oChannel;
        this.initLayers();
    }

    public DiTSimpleHeadLayer(int iChannel, int oChannel, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.iChannel = iChannel;
        this.oChannel = oChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = oChannel;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(iChannel, iChannel + oChannel, bias, network);
        this.linear1.weight.setData(RandomUtils.xavierUniform(iChannel * (iChannel + oChannel), iChannel, (iChannel + oChannel), 1.0f));
        if(this.linear1.bias != null) {
        	this.linear1.bias.clearGPU();
        }
        this.active = new GeluLayer(linear1);
        this.linear2 = new FullyLayer((iChannel + oChannel), oChannel, bias, network);
        this.linear2.weight.setData(RandomUtils.xavierUniform((iChannel + oChannel) * oChannel, (iChannel + oChannel), oChannel, 1.0f));
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
        linear1.forward(input);
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
    	this.linear2.back(this.delta);
        active.back(this.linear2.diff);
        linear1.back(active.diff);
        this.diff = this.linear1.diff;
        //		System.out.println("mlp diff:");
        //		diff.showDMByNumber(0);
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }
}

