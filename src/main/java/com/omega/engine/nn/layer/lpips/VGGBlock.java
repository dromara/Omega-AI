package com.omega.engine.nn.layer.lpips;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * resnet block layer
 *
 * @author Administrator
 */
public class VGGBlock extends Layer {
    public ConvolutionLayer conv;
    private boolean bn = false;
    private boolean freeze = false;
    private BNLayer norm;
    private ReluLayer act;

    public VGGBlock(int channel, int oChannel, int height, int width, boolean bn, boolean freeze, Network network) {
        this.network = network;
        this.bn = bn;
        this.freeze = freeze;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public void initLayers() {
        conv = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, freeze, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.relu;
        if (bn) {
            norm = new BNLayer(conv);
            norm.freeze();
            act = new ReluLayer(norm);
        } else {
            act = new ReluLayer(conv);
        }
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
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
        conv.forward(x);
        x = conv.getOutput();
        if (bn) {
            norm.forward(x);
            x = norm.getOutput();
        }
        act.forward(x);
        this.output = act.getOutput();
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
        act.back(diffOut);
        if (bn) {
            norm.back(diffOut);
            diffOut = norm.diff;
        }
        conv.back(diffOut);
        this.diff = conv.diff;
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
        if (bn) {
            norm.update();
        }
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
        if (bn) {
            norm.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
        if (bn) {
            norm.loadModel(inputStream);
        }
    }
}

