package com.omega.engine.nn.layer.vae.tiny;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
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
public class TinyVAEConvTransposeBlock extends Layer {
    //	private int group = 32;
    public ConvolutionTransposeLayer conv;
    //	private GNLayer norm;
    public BNLayer norm;
    private ActiveFunctionLayer act;

    public TinyVAEConvTransposeBlock(int channel, int oChannel, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
    }

    public void initLayers() {
        conv = new ConvolutionTransposeLayer(channel, oChannel, width, height, 3, 3, 1, 2, 1, 1, true, network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.leaky_relu;
        norm = new BNLayer(conv);
        act = new LeakyReluLayer(norm);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        //		if(this.output == null || this.output.number != this.network.number) {
        //			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        //		}
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
        conv.forward(this.input);
        norm.forward(conv.getOutput());
        act.forward(norm.getOutput());
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
        act.back(this.delta);
        norm.back(act.diff);
        conv.back(norm.diff);
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
        norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
        norm.loadModel(inputStream);
    }
}

