package com.omega.engine.nn.layer.clip;

import com.omega.common.tensor.Tensor;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * PoswiseFeedForward Layer
 *
 * @author Administrator
 */
public class CLIPMLPLayer extends Layer {
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private FullyLayer linear1;
    private GeluLayer active;
    private FullyLayer linear2;

    public CLIPMLPLayer(int embedDim, int nChannel, boolean bias) {
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public CLIPMLPLayer(int embedDim, int nChannel, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);
        this.active = new GeluLayer(getLinear1(), true);
        this.linear2 = new FullyLayer(nChannel, embedDim, bias, network);
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
        if (network.RUN_MODEL == RunModel.EVAL) {
            Tensor cache = CUDAMemoryManager.getCache("CLIIP_mlp_cache", input.number, 1, 1, nChannel);
            Tensor cache2 = CUDAMemoryManager.getCache("CLIIP_mlp_cache2", input.number, 1, 1, embedDim);
            getLinear1().forward(input, cache);
            active.forward(getLinear1().getOutput(), cache);
            getLinear2().forward(active.getOutput(), cache2);
        } else {
            getLinear1().forward(input);
            active.forward(getLinear1().getOutput());
            getLinear2().forward(active.getOutput());
        }
        this.output = getLinear2().getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
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
        getLinear1().update();
        getLinear2().update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.clip_mlp;
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
        getLinear1().saveModel(outputStream);
        getLinear2().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getLinear1().loadModel(inputStream);
        getLinear2().loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        getLinear1().accGrad(scale);
        getLinear2().accGrad(scale);
    }

    public FullyLayer getLinear1() {
        return linear1;
    }

    public FullyLayer getLinear2() {
        return linear2;
    }
}

