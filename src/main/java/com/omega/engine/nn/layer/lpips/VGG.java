package com.omega.engine.nn.layer.lpips;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.pooling.PoolingType;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * VGG
 *
 * @author Administrator
 */
public class VGG extends Layer {
    public List<Layer> features;
    private boolean bn = false;
    private boolean freeze = false;
    private boolean onlyFeatures = false;
    private String[] cfg;
    private int numClass = 1000;
    private AdaptiveAvgPool2DLayer avgpool;
    private FullyLayer c1;
    private ReluLayer a1;
    private FullyLayer c2;
    private ReluLayer a2;
    private FullyLayer co;

    public VGG(int channel, int numClass, int height, int width, boolean bn, boolean freeze, String[] cfg, Network network) {
        this.network = network;
        this.cfg = cfg;
        this.bn = bn;
        this.freeze = freeze;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.numClass = numClass;
        initLayers();
    }

    public VGG(int channel, int numClass, int height, int width, boolean bn, String[] cfg, boolean onlyFeatures, boolean freeze, Network network) {
        this.network = network;
        this.cfg = cfg;
        this.bn = bn;
        this.freeze = freeze;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.numClass = numClass;
        this.onlyFeatures = onlyFeatures;
        initLayers();
    }

    public void initLayers() {
        int ic = 3;
        int ih = height;
        int iw = width;
        features = new ArrayList<Layer>();
        for (String v : cfg) {
            if (v.equals("M")) {
                PoolingLayer maxpool = new PoolingLayer(ic, iw, ih, 2, 2, 2, PoolingType.MAX_POOLING, network);
                features.add(maxpool);
                ih = maxpool.oHeight;
                iw = maxpool.oWidth;
            } else {
                int oc = Integer.parseInt(v);
                VGGBlock b = new VGGBlock(ic, oc, ih, iw, bn, freeze, network);
                features.add(b);
                ic = b.oChannel;
                ih = b.oHeight;
                iw = b.oWidth;
            }
        }
        if (!onlyFeatures) {
            avgpool = new AdaptiveAvgPool2DLayer(ic, ih, iw, 7, 7);
            c1 = new FullyLayer(512 * 7 * 7, 4096, true, network);
            a1 = new ReluLayer(c1);
            c2 = new FullyLayer(4096, 4096, true, network);
            a2 = new ReluLayer(c2);
            co = new FullyLayer(4096, numClass, true, network);
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
        for (int i = 0; i < features.size(); i++) {
            Layer l = features.get(i);
            l.forward(x);
            x = l.getOutput();
        }
        if (avgpool.getOutput() != null) {
            avgpool.getOutput().viewOrg();
        }
        avgpool.forward(x);
        x = avgpool.getOutput().view(avgpool.number, 1, 1, avgpool.oChannel * avgpool.oHeight * avgpool.oWidth);
        c1.forward(x);
        a1.forward(c1.getOutput());
        c2.forward(a1.getOutput());
        a2.forward(c2.getOutput());
        co.forward(a2.getOutput());
        this.output = co.getOutput();
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
        co.back(diffOut);
        a2.back(co.diff);
        c2.back(a2.diff);
        a1.back(c2.diff);
        c1.back(a1.diff);
        avgpool.back(c1.diff);
        diffOut = avgpool.diff;
        for (int i = features.size() - 1; i >= 0; i--) {
            Layer l = features.get(i);
            l.forward(diffOut);
            diffOut = l.diff;
        }
        this.diff = diffOut;
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

    public Tensor[] features(Tensor input, int[] featuresIndex) {
        Tensor[] fr = new Tensor[featuresIndex.length];
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        Tensor x = input;
        int index = 0;
        for (int i = 0; i < features.size(); i++) {
            Layer l = features.get(i);
            l.forward(x);
            x = l.getOutput();
            if (i == featuresIndex[index]) {
                fr[index] = l.getOutput();
                index++;
            }
        }
        return fr;
    }

    public void features(Tensor input, int[] featuresIndex, Tensor[] fr) {
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        Tensor x = input;
        int index = 0;
        for (int i = 0; i < features.size(); i++) {
            Layer l = features.get(i);
            l.forward(x);
            x = l.getOutput();
            if (index < featuresIndex.length && i == featuresIndex[index]) {
                fr[index] = x;
                index++;
            }
        }
    }

    public void featuresCopy(Tensor input, int[] featuresIndex, Tensor[] fr) {
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        Tensor x = input;
        int index = 0;
        for (int i = 0; i < features.size(); i++) {
            Layer l = features.get(i);
            l.forward(x);
            x = l.getOutput();
            if (index < featuresIndex.length && i == featuresIndex[index]) {
                x.copyGPU(fr[index]);
                index++;
            }
        }
    }

    public Tensor featuresBackward(Tensor[] deltas, int[] featuresIndex) {
        initBack();
        Tensor d = deltas[deltas.length - 1];
        int index = featuresIndex.length - 2;
        for (int i = features.size() - 1; i >= 0; i--) {
            Layer l = features.get(i);
            l.back(d);
            d = l.diff;
            if (i > 0 && index >= 0 && i - 1 == featuresIndex[index]) {
                Tensor_OP().add(d, deltas[index], d);
                index--;
            }
        }
        //		d.showDM();
        return d;
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
        for (int i = 0; i < features.size(); i++) {
            features.get(i).update();
        }
        c1.update();
        c2.update();
        co.update();
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        //		conv.loadModel(inputStream);
        //
        //		if(bn) {
        //			norm.loadModel(inputStream);
        //		}
    }
}

