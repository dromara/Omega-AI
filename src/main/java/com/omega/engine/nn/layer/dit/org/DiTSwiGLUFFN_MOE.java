package com.omega.engine.nn.layer.dit.org;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.DiTMoEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Top-1 MoE version of DiTSwiGLUFFN.
 *
 * The forward/backward path uses compact top-1 token dispatch so only the
 * routed expert runs for each token. Backward recomputes expert activations to
 * avoid storing per-expert full-batch intermediates.
 */
public class DiTSwiGLUFFN_MOE extends Layer {

    private int inChannel = 0;
    private int hiddenSize;
    private int outChannel;
    private int numExperts;
    private float auxLossCoef = 5.0e-4f;

    private boolean bias = false;

    private FullyLayer gate;
    public FullyLayer[] w12;
    public FullyLayer[] w3;

    private Tensor routeProb;
    private Tensor topIdx;
    private Tensor topSlot;
    private Tensor topWeight;
    private Tensor topWeightGrad;
    private Tensor expertCounts;
    private Tensor gateDelta;
    private Tensor auxLoss;

    private Tensor diffCache;

    private Tensor expertInput;
    private Tensor expertOutput;
    private Tensor expertDelta;
    private Tensor expertDiff;
    private Tensor w12Output;
    private Tensor w12Delta;
    private Tensor w1;
    private Tensor w2;
    private Tensor act;
    private Tensor wt;
    private Tensor dwt;

    private float[] expertCountsHost;

    private DiTMoEKernel moeKernel;

    public DiTSwiGLUFFN_MOE(int inChannel, int hiddenSize, int outChannel, boolean bias) {
        this(inChannel, hiddenSize, outChannel, bias, 4, 5.0e-4f);
    }

    public DiTSwiGLUFFN_MOE(int inChannel, int hiddenSize, int outChannel, boolean bias, int numExperts, float auxLossCoef) {
        this.inChannel = inChannel;
        this.hiddenSize = hiddenSize;
        this.outChannel = outChannel;
        this.bias = bias;
        this.numExperts = numExperts;
        this.auxLossCoef = auxLossCoef;
        this.channel = 1;
        this.height = 1;
        this.width = inChannel;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = outChannel;
        this.initLayers();
    }

    public DiTSwiGLUFFN_MOE(int inChannel, int hiddenSize, int outChannel, boolean bias, Network network) {
        this(inChannel, hiddenSize, outChannel, bias, 4, 5.0e-4f, network);
    }

    public DiTSwiGLUFFN_MOE(int inChannel, int hiddenSize, int outChannel, boolean bias, int numExperts, float auxLossCoef, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.inChannel = inChannel;
        this.hiddenSize = hiddenSize;
        this.outChannel = outChannel;
        this.bias = bias;
        this.numExperts = numExperts;
        this.auxLossCoef = auxLossCoef;
        this.channel = 1;
        this.height = 1;
        this.width = inChannel;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = outChannel;
        this.initLayers();
    }

    public void initLayers() {
        this.gate = new FullyLayer(inChannel, numExperts, true, network);
        RandomUtils.xavier_uniform(gate.weight, 1, inChannel, numExperts);
        if (gate.bias != null) {
            gate.bias.clearGPU();
        }

        this.w12 = new FullyLayer[numExperts];
        this.w3 = new FullyLayer[numExperts];

        for (int i = 0; i < numExperts; i++) {
            this.w12[i] = new FullyLayer(inChannel, hiddenSize * 2, bias, network);
            RandomUtils.xavier_uniform(w12[i].weight, 1, inChannel, hiddenSize * 2);
            if (w12[i].bias != null) {
                w12[i].bias.clearGPU();
            }
            this.w3[i] = new FullyLayer(hiddenSize, outChannel, bias, network);
            RandomUtils.xavier_uniform(w3[i].weight, 1, hiddenSize, outChannel);
            if (w3[i].bias != null) {
                w3[i].bias.clearGPU();
            }
        }

        if (moeKernel == null) {
            moeKernel = new DiTMoEKernel(cuda());
        }
    }

    @Override
    public void init() {
        this.number = this.input.number;
        initTensor(number);
    }

    public void init(Tensor input) {
        this.number = input.number;
        initTensor(number);
    }

    private void initTensor(int number) {
        if (output == null || output.number != number) {
            output = Tensor.createGPUTensor(output, number, 1, 1, outChannel, true);
            diffCache = Tensor.createGPUTensor(diffCache, number, 1, 1, inChannel, true);
            routeProb = Tensor.createGPUTensor(routeProb, number, 1, 1, numExperts, true);
            topIdx = Tensor.createGPUTensor(topIdx, number, 1, 1, 1, true);
            topSlot = Tensor.createGPUTensor(topSlot, number, 1, 1, 1, true);
            topWeight = Tensor.createGPUTensor(topWeight, number, 1, 1, 1, true);
            topWeightGrad = Tensor.createGPUTensor(topWeightGrad, number, 1, 1, 1, true);
            expertCounts = Tensor.createGPUTensor(expertCounts, 1, 1, 1, numExperts, true);
            gateDelta = Tensor.createGPUTensor(gateDelta, number, 1, 1, numExperts, true);
            auxLoss = Tensor.createGPUTensor(auxLoss, 1, 1, 1, 1, true);

            expertInput = Tensor.createGPUTensor(expertInput, number, 1, 1, inChannel, true);
            expertOutput = Tensor.createGPUTensor(expertOutput, number, 1, 1, outChannel, true);
            expertDelta = Tensor.createGPUTensor(expertDelta, number, 1, 1, outChannel, true);
            expertDiff = Tensor.createGPUTensor(expertDiff, number, 1, 1, inChannel, true);
            w12Output = Tensor.createGPUTensor(w12Output, number, 1, 1, hiddenSize * 2, true);
            w12Delta = Tensor.createGPUTensor(w12Delta, number, 1, 1, hiddenSize * 2, true);
            w1 = Tensor.createGPUTensor(w1, number, 1, 1, hiddenSize, true);
            w2 = Tensor.createGPUTensor(w2, number, 1, 1, hiddenSize, true);
            act = Tensor.createGPUTensor(act, number, 1, 1, hiddenSize, true);
            wt = Tensor.createGPUTensor(wt, number, 1, 1, hiddenSize, true);
            dwt = Tensor.createGPUTensor(dwt, number, 1, 1, hiddenSize, true);
        }
        if (expertCountsHost == null || expertCountsHost.length != numExperts) {
            expertCountsHost = new float[numExperts];
        }
    }

    @Override
    public void initBack() {
        for (int i = 0; i < numExperts; i++) {
            initParamGrad(w12[i]);
            initParamGrad(w3[i]);
        }
    }

    @Override
    public void initParam() {
    }

    @Override
    public void output() {
        gate.forward(input);
        moeKernel.routeTop1(gate.getOutput(), routeProb, topIdx, topSlot, topWeight, expertCounts, number, numExperts);
        moeKernel.auxLoss(routeProb, expertCounts, auxLoss, number, numExperts, auxLossCoef);
        expertCounts.syncHost(expertCountsHost);

        output.clearGPU();
        for (int i = 0; i < numExperts; i++) {
            int count = expertCount(i);
            if (count <= 0) {
                continue;
            }
            forwardExpert(i, count);
            moeKernel.combineSparseForward(expertOutput, output, topIdx, topSlot, topWeight, i, number, outChannel);
        }
    }

    private void forwardExpert(int i, int count) {
        viewExpertBuffers(count);
        moeKernel.dispatchForward(input, expertInput, topIdx, topSlot, i, number, inChannel);
        linearForward(w12[i], expertInput, w12Output, count, inChannel, hiddenSize * 2);
        moeKernel.swigluForward(w12Output, w1, w2, act, wt, count, hiddenSize);
        linearForward(w3[i], wt, expertOutput, count, hiddenSize, outChannel);
    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    public Tensor getAuxLoss() {
        return auxLoss;
    }

    public Tensor getExpertCounts() {
        return expertCounts;
    }

    public Tensor getRouteProb() {
        return routeProb;
    }

    @Override
    public void diff() {
        topWeightGrad.clearGPU();
        diffCache.clearGPU();

        for (int i = 0; i < numExperts; i++) {
            int count = expertCount(i);
            if (count <= 0) {
                clearParamGrad(w12[i]);
                clearParamGrad(w3[i]);
                continue;
            }
            forwardExpert(i, count);
            moeKernel.sparseExpertDelta(delta, expertOutput, expertDelta, topWeightGrad, topIdx, topSlot, topWeight, i, number, outChannel);
            backwardExpert(i, count);
            moeKernel.scatterAddDiff(expertDiff, diffCache, topIdx, topSlot, i, number, inChannel);
        }

        moeKernel.gateBackward(routeProb, topIdx, topWeightGrad, expertCounts, gateDelta, number, numExperts, auxLossCoef);
        gate.back(gateDelta);
        Tensor_OP().add(diffCache, gate.diff, diffCache);
        this.diff = diffCache;
    }

    private void backwardExpert(int i, int count) {
        linearBackward(w3[i], wt, expertDelta, dwt, count, hiddenSize, outChannel);
        moeKernel.swigluBackward(dwt, w1, w2, act, w12Delta, count, hiddenSize);
        linearBackward(w12[i], expertInput, w12Delta, expertDiff, count, inChannel, hiddenSize * 2);
    }

    private void viewExpertBuffers(int count) {
        expertInput.view(count, 1, 1, inChannel);
        expertOutput.view(count, 1, 1, outChannel);
        expertDelta.view(count, 1, 1, outChannel);
        expertDiff.view(count, 1, 1, inChannel);
        w12Output.view(count, 1, 1, hiddenSize * 2);
        w12Delta.view(count, 1, 1, hiddenSize * 2);
        w1.view(count, 1, 1, hiddenSize);
        w2.view(count, 1, 1, hiddenSize);
        act.view(count, 1, 1, hiddenSize);
        wt.view(count, 1, 1, hiddenSize);
        dwt.view(count, 1, 1, hiddenSize);
    }

    private int expertCount(int expertId) {
        int count = Math.round(expertCountsHost[expertId]);
        if (count < 0) {
            return 0;
        }
        return Math.min(count, number);
    }

    private void linearForward(FullyLayer layer, Tensor in, Tensor out, int rows, int inputWidth, int outputWidth) {
        GPU_OP().multiplyFloat(CUBLAS_OP_N, CUBLAS_OP_T, rows, outputWidth, inputWidth, 1,
                in.getGpuData(), inputWidth, layer.weight.getGpuData(), inputWidth, 0, out.getGpuData(), outputWidth);
        if (layer.bias != null) {
            moeKernel.addBias(out, layer.bias, rows, outputWidth);
        }
    }

    private void linearBackward(FullyLayer layer, Tensor in, Tensor dOut, Tensor dIn, int rows, int inputWidth, int outputWidth) {
        initParamGrad(layer);
        GPU_OP().multiplyFloat(CUBLAS_OP_T, CUBLAS_OP_N, outputWidth, inputWidth, rows, 1,
                dOut.getGpuData(), outputWidth, in.getGpuData(), inputWidth, 0, layer.diffW.getGpuData(), inputWidth);
        if (layer.PROPAGATE_DOWN) {
            GPU_OP().multiplyFloat(CUBLAS_OP_N, CUBLAS_OP_N, rows, inputWidth, outputWidth, 1,
                    dOut.getGpuData(), outputWidth, layer.weight.getGpuData(), inputWidth, 0, dIn.getGpuData(), inputWidth);
        }
        if (layer.bias != null) {
            moeKernel.biasBackward(dOut, layer.diffB, rows, outputWidth);
        }
    }

    private void initParamGrad(FullyLayer layer) {
        if (layer.diffW == null) {
            layer.diffW = new Tensor(1, 1, layer.oWidth, layer.width, true, true);
        }
        if (layer.bias != null && layer.diffB == null) {
            layer.diffB = new Tensor(1, 1, 1, layer.oWidth, true);
        }
    }

    private void clearParamGrad(FullyLayer layer) {
        initParamGrad(layer);
        layer.diffW.clearGPU();
        if (layer.diffB != null) {
            layer.diffB.clearGPU();
        }
    }

    @Override
    public void forward() {
        this.setInput();
        this.init();
        this.output();
    }

    @Override
    public void back() {
        this.initBack();
        this.setDelta();
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        this.setInput(input);
        this.init();
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        this.initBack();
        this.setDelta(delta);
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        gate.update();
        for (int i = 0; i < numExperts; i++) {
            w12[i].update();
            w3[i].update();
        }
    }

    @Override
    public void showDiff() {
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        gate.saveModel(outputStream);
        for (int i = 0; i < numExperts; i++) {
            w12[i].saveModel(outputStream);
            w3[i].saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        gate.loadModel(inputStream);
        for (int i = 0; i < numExperts; i++) {
            w12[i].loadModel(inputStream);
            w3[i].loadModel(inputStream);
        }
    }

    @Override
    public void accGrad(float scale) {
        gate.accGrad(scale);
        for (int i = 0; i < numExperts; i++) {
            w12[i].accGrad(scale);
            w3[i].accGrad(scale);
        }
    }
}
