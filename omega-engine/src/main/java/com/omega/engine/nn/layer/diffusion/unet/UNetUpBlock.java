package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.diffusion.UpSampleLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;
import java.util.Stack;

/**
 * UNetUpBlock
 *
 * @author Administrator
 */
public class UNetUpBlock extends Layer {
    public UpSampleLayer up;
    public UNetResidualBlock[] res;
    public UNetAttentionBlock[] attns;
    private int catChannel;
    private int numLayer = 1;
    private int headNum = 4;
    private int contextTime = 0;
    private int contextDim = 0;
    private int nTime = 0;
    private int groupNum = 32;
    private RouteLayer[] cats;

    public UNetUpBlock(int catChannel, int channel, int oChannel, int height, int width, int nTime, int headNum, int contextTime, int contextDim, int groupNum, int numLayer, Stack<Layer> downLayers, Network network) {
        this.network = network;
        this.nTime = nTime;
        this.numLayer = numLayer;
        this.groupNum = groupNum;
        this.headNum = headNum;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.catChannel = catChannel;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers(downLayers);
    }

    public UNetUpBlock(String name, int catChannel, int channel, int oChannel, int height, int width, int nTime, int headNum, int contextTime, int contextDim, int groupNum, int numLayer, Stack<Layer> downLayers, Network network) {
        this.setName(name);
        this.network = network;
        this.nTime = nTime;
        this.numLayer = numLayer;
        this.groupNum = groupNum;
        this.headNum = headNum;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.catChannel = catChannel;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers(downLayers);
    }

    public static void loadWeight(Map<String, Object> weightMap, UNetUpBlock network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
    }

    public static void main(String[] args) {
    }

    public void initLayers(Stack<Layer> downLayers) {
        int ic = catChannel + channel;
        up = new UpSampleLayer(ic, channel, height, width, network);
        up.setName("[" + name + "]up-upsample");
        this.oHeight = up.oHeight;
        this.oWidth = up.oWidth;
        res = new UNetResidualBlock[numLayer];
        attns = new UNetAttentionBlock[numLayer];
        cats = new RouteLayer[numLayer];
        for (int i = 0; i < numLayer; i++) {
            int oc = channel;
            if (i == numLayer - 1) {
                oc = oChannel;
            }
            UNetResidualBlock rb = new UNetResidualBlock(ic, oc, oHeight, oWidth, nTime, groupNum, network);
            UNetAttentionBlock attn = new UNetAttentionBlock(oc, oHeight, oWidth, headNum, contextTime, contextDim, groupNum, network);
            attn.setName("[" + name + "]up-attn-" + i);
            res[i] = rb;
            attns[i] = attn;
        }
        Layer dc0 = downLayers.pop();
        cats[0] = new RouteLayer(new Layer[]{up, dc0});
        for (int i = numLayer - 1; i > 0; i--) {
            Layer dc = downLayers.pop();
            cats[i] = new RouteLayer(new Layer[]{attns[i - 1], dc});
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
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
    }

    public void output(Tensor time) {
        // TODO Auto-generated method stub
        up.forward(input);
        Tensor x = null;
        for (int i = 0; i < numLayer; i++) {
            cats[i].forward();
            res[i].forward(cats[i].getOutput(), time);
            attns[i].forward(res[i].getOutput());
            x = attns[i].getOutput();
        }
        this.output = x;
    }

    public void output(Tensor time, Tensor context) {
        // TODO Auto-generated method stub
        up.forward(input);
        Tensor x = null;
        for (int i = 0; i < numLayer; i++) {
            cats[i].forward();
            res[i].forward(cats[i].getOutput(), time);
            attns[i].forward(res[i].getOutput(), context);
            x = attns[i].getOutput();
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
    }

    public void diff(Tensor timeDiff) {
        // TODO Auto-generated method stub
        Tensor d = delta;
        for (int i = numLayer - 1; i >= 0; i--) {
            //			d.showDM("up-attn-delta");
            attns[i].back(d);
            //			attns[i].diff.showDM("up-attn");
            res[i].back(attns[i].diff, timeDiff);
            cats[i].back(res[i].diff);
            if (i > 0) {
                d = attns[i - 1].cache_delta;
            }
        }
        up.back(up.cache_delta);
        this.diff = up.diff;
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
    public void backTemp() {
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

    public void forward(Tensor input, Tensor time) {
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
        this.output(time);
    }

    public void forward(Tensor input, Tensor time, Tensor context) {
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
        this.output(time, context);
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

    public void back(Tensor delta, Tensor timeDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(timeDiff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        up.update();
        for (int i = 0; i < numLayer; i++) {
            res[i].update();
            attns[i].update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
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
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        up.accGrad(scale);
        for (int i = 0; i < numLayer; i++) {
            res[i].accGrad(scale);
            attns[i].accGrad(scale);
        }
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        up.saveModel(outputStream);
        for (int i = 0; i < numLayer; i++) {
            res[i].saveModel(outputStream);
            attns[i].saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        up.loadModel(inputStream);
        for (int i = 0; i < numLayer; i++) {
            res[i].loadModel(inputStream);
            attns[i].loadModel(inputStream);
        }
    }
}

