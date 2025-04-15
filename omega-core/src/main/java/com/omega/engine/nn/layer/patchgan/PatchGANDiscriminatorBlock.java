package com.omega.engine.nn.layer.patchgan;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

public class PatchGANDiscriminatorBlock extends Layer {
    public List<Layer> layers;
    private int[] convChannels;
    private int[] kernels;
    private int[] strides;
    private int[] paddings;

    public PatchGANDiscriminatorBlock(int channel, int height, int width, int[] convChannels, int[] kernels, int[] strides, int[] paddings, Network network) {
        this.network = network;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = 1;
        this.convChannels = convChannels;
        this.kernels = kernels;
        this.paddings = paddings;
        this.strides = strides;
        initLayers();
    }

    public void initLayers() {
        layers = new ArrayList<Layer>();
        int ih = height;
        int iw = width;
        for (int i = 0; i < convChannels.length - 1; i++) {
            boolean hasBias = false;
            if (i == 0) {
                hasBias = true;
            }
            ConvolutionLayer conv = new ConvolutionLayer(convChannels[i], convChannels[i + 1], iw, ih, kernels[i], kernels[i], paddings[i], strides[i], hasBias, network);
            conv.setUpdater(UpdaterFactory.create(this.network));
            conv.paramsInit = ParamsInit.leaky_relu;
            layers.add(conv);
            Layer next = conv;
            if (i != convChannels.length - 2 && i != 0) {
                BNLayer bn = new BNLayer(next);
                layers.add(bn);
                next = bn;
            }
            if (i != convChannels.length - 2) {
                LeakyReluLayer act = new LeakyReluLayer(next);
                layers.add(act);
            }
            ih = conv.oHeight;
            iw = conv.oWidth;
        }
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
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.forward(x);
            x = layer.getOutput();
        }
        this.output = x;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        Tensor d = this.delta;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            layer.back(d);
            d = layer.diff;
        }
        this.diff = d;
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
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.lpips;
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    }
}

