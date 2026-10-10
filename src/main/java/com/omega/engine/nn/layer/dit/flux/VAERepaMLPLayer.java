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
 * Five-layer projection head used by the experimental VAE-REPA model.
 */
public class VAERepaMLPLayer extends Layer {

    private final int embedDim;
    private final int projectorDim;
    private final int latentDim;
    private final boolean bias;

    public FullyLayer linear1;
    public FullyLayer linear2;
    public FullyLayer linear3;
    public FullyLayer linear4;
    public FullyLayer linear5;

    private SiLULayer active1;
    private SiLULayer active2;
    private SiLULayer active3;
    private SiLULayer active4;

    public VAERepaMLPLayer(int embedDim, int projectorDim, int latentDim, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.projectorDim = projectorDim;
        this.latentDim = latentDim;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = latentDim;
        initLayers();
    }

    private FullyLayer linear(int inputDim, int outputDim) {
        FullyLayer layer = new FullyLayer(inputDim, outputDim, bias, network);
        layer.weight.setData(RandomUtils.xavierUniform(inputDim * outputDim, inputDim, outputDim, 1.0f));
        if (layer.bias != null) {
            layer.bias.clearGPU();
        }
        return layer;
    }

    private void initLayers() {
        linear1 = linear(embedDim, projectorDim);
        active1 = new SiLULayer(linear1);
        linear2 = linear(projectorDim, projectorDim);
        active2 = new SiLULayer(linear2);
        linear3 = linear(projectorDim, projectorDim);
        active3 = new SiLULayer(linear3);
        linear4 = linear(projectorDim, projectorDim);
        active4 = new SiLULayer(linear4);
        linear5 = linear(projectorDim, latentDim);
    }

    @Override
    public void init() {
        this.number = this.input.number;
    }

    @Override
    public void initBack() {
    }

    @Override
    public void initParam() {
    }

    @Override
    public void output() {
        linear1.forward(input);
        active1.forward(linear1.getOutput());
        linear2.forward(active1.getOutput());
        active2.forward(linear2.getOutput());
        linear3.forward(active2.getOutput());
        active3.forward(linear3.getOutput());
        linear4.forward(active3.getOutput());
        active4.forward(linear4.getOutput());
        linear5.forward(active4.getOutput());
        this.output = linear5.getOutput();
    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {
        linear5.back(delta);
        active4.back(linear5.diff);
        linear4.back(active4.diff);
        active3.back(linear4.diff);
        linear3.back(active3.diff);
        active2.back(linear3.diff);
        linear2.back(active2.diff);
        active1.back(linear2.diff);
        linear1.back(active1.diff);
        this.diff = linear1.diff;
    }

    @Override
    public void forward() {
        setInput();
        init();
        output();
    }

    @Override
    public void forward(Tensor input) {
        setInput(input);
        init();
        output();
    }

    @Override
    public void back() {
        initBack();
        setDelta();
        diff();
        if (network.GRADIENT_CHECK) {
            gradientCheck();
        }
    }

    @Override
    public void back(Tensor delta) {
        initBack();
        setDelta(delta);
        diff();
        if (network.GRADIENT_CHECK) {
            gradientCheck();
        }
    }

    @Override
    public void update() {
        linear1.update();
        linear2.update();
        linear3.update();
        linear4.update();
        linear5.update();
    }

    @Override
    public void accGrad(float scale) {
        linear1.accGrad(scale);
        linear2.accGrad(scale);
        linear3.accGrad(scale);
        linear4.accGrad(scale);
        linear5.accGrad(scale);
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.mlp;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        return null;
    }

    @Override
    public void initCache() {
    }

    @Override
    public void backTemp() {
    }

    @Override
    public void showDiff() {
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
        linear3.saveModel(outputStream);
        linear4.saveModel(outputStream);
        linear5.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
        linear3.loadModel(inputStream);
        linear4.loadModel(inputStream);
        linear5.loadModel(inputStream);
    }
}
