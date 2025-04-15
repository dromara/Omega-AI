package com.omega.engine.nn.layer.tae;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * resnet block layer
 *
 * @author Administrator
 */
public class TAEDecoder extends Layer {
    private int upsamplingScalingFactor = 2;
    private int[] numBlocks;
    private int[] blockOutChannels;
    private List<Layer> layers;

    public TAEDecoder(int channel, int oChannel, int height, int width, int upsamplingScalingFactor, int[] numBlocks, int[] blockOutChannels, Network network) {
        this.network = network;
        this.upsamplingScalingFactor = upsamplingScalingFactor;
        this.numBlocks = numBlocks;
        this.blockOutChannels = blockOutChannels;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = layers.get(layers.size() - 1).oHeight;
        this.oWidth = layers.get(layers.size() - 1).oWidth;
    }

    public void initLayers() {
        layers = new ArrayList<Layer>();
        ConvolutionLayer head = new ConvolutionLayer(channel, blockOutChannels[0], width, height, 3, 3, 1, 1, true, this.network);
        head.setUpdater(UpdaterFactory.create(this.network));
        head.paramsInit = ParamsInit.relu;
        ReluLayer headAct = new ReluLayer(head);
        layers.add(head);
        layers.add(headAct);
        int numChannels = 0;
        int ih = height;
        int iw = width;
        for (int i = 0; i < numBlocks.length; i++) {
            numChannels = blockOutChannels[i];
            for (int j = 0; j < numBlocks[i]; j++) {
                TAEBlock tb = new TAEBlock(numChannels, numChannels, ih, iw, network);
                layers.add(tb);
            }
            int convOutChannel = numChannels;
            boolean hasBias = false;
            if (i == numBlocks.length - 1) {
                UPSampleLayer2 up = new UPSampleLayer2(numChannels, ih, iw, upsamplingScalingFactor, network);
                layers.add(up);
                ih = up.oHeight;
                iw = up.oWidth;
                convOutChannel = oChannel;
                hasBias = true;
            }
            ConvolutionLayer bconv = new ConvolutionLayer(numChannels, convOutChannel, width, height, 3, 3, 1, 1, hasBias, this.network);
            bconv.setUpdater(UpdaterFactory.create(this.network));
            bconv.paramsInit = ParamsInit.relu;
            layers.add(bconv);
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
        Tensor dx = this.delta;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            layer.back(dx);
            dx = layer.diff;
        }
        this.diff = dx;
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
            ;
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
}

