package com.omega.engine.nn.layer.llama;

import com.omega.common.tensor.Tensor;
import com.omega.utils.RandomUtils;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * PoswiseFeedForward Layer
 *
 * @author Administrator
 */
public class LlamaMLPLayer extends Layer {
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private boolean dropout = false;
    private FullyLayer linear1;
    private SiLULayer active;
    private FullyLayer linear2;
    private FullyLayer linear3;
    private DropoutLayer dropoutLayer;
    private Tensor tmp;
    private int multiple_of = 64;

    public LlamaMLPLayer(int embedDim, int nChannel, boolean bias) {
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.nChannel = 4 * this.embedDim;
        this.nChannel = (int) (2 * this.nChannel / 3);
        this.nChannel = multiple_of * ((this.nChannel + this.multiple_of - 1) / this.multiple_of);
        this.initLayers();
    }

    public LlamaMLPLayer(int embedDim, int nChannel, int multiple_of, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.multiple_of = multiple_of;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.nChannel = 4 * this.embedDim;
        this.nChannel = (int) (2 * this.nChannel / 3);
        this.nChannel = multiple_of * ((this.nChannel + this.multiple_of - 1) / this.multiple_of);
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        //		NanoGPT net = (NanoGPT) this.network;
        this.setLinear1(new FullyLayer(embedDim, nChannel, bias, network));
        this.linear1.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.normal_(this.embedDim * nChannel, 0.01f, 0.02f), true);
        //		this.linear1.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.order(this.embedDim * nChannel, 0.001f, 0.001f), true);
        //		Tensor qw = new Tensor(1, 1, embedDim, nChannel, true);
        //		Tensor_OP().permute(this.linear1.weight, qw, new int[] {0, 1, 3, 2});
        //		this.linear1.weight = qw;
        //		this.linear1.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, 0.02f), true);
        this.active = new SiLULayer(getLinear1());
        this.setLinear2(new FullyLayer(nChannel, embedDim, bias, network));
        this.linear2.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.normal_(this.nChannel * embedDim, 0.01f, 0.02f), true);
        //		this.linear2.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.order(this.nChannel * embedDim, 0.001f, 0.001f), true);
        //		Tensor w2 = new Tensor(1, 1, nChannel, embedDim, true);
        //		Tensor_OP().permute(this.linear2.weight, w2, new int[] {0, 1, 3, 2});
        //		this.linear2.weight = w2;
        //		this.linear2.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, 0.02f), true);
        //		this.linear2.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, (0.02f / (float) Math.sqrt(2 * net.decoderNum))), true);
        this.setLinear3(new FullyLayer(embedDim, nChannel, bias, network));
        this.linear3.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.normal_(this.embedDim * nChannel, 0.01f, 0.02f), true);
        //		this.linear3.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.order(this.embedDim * nChannel, 0.001f, 0.001f), true);
        //		Tensor w3 = new Tensor(1, 1, embedDim, nChannel, true);
        //		Tensor_OP().permute(this.linear3.weight, w3, new int[] {0, 1, 3, 2});
        //		this.linear3.weight = w3;
        if (dropout) {
            dropoutLayer = new DropoutLayer(0.1f, getLinear2());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if (this.tmp == null || this.number != this.tmp.number) {
            this.tmp = Tensor.createGPUTensor(this.tmp, number, oChannel, oHeight, nChannel, true);
        }
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
        getLinear1().forward(input);
        active.forward(getLinear1().getOutput());
        getLinear3().forward(input);
        Tensor_OP().mul(active.getOutput(), getLinear3().getOutput(), tmp);
        getLinear2().forward(tmp);
        if (dropout) {
            dropoutLayer.forward(getLinear2().getOutput());
            this.output = dropoutLayer.getOutput();
        } else {
            this.output = getLinear2().getOutput();
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        if (this.dropout) {
            this.dropoutLayer.back(delta);
            this.getLinear2().back(this.dropoutLayer.diff);
        } else {
            this.getLinear2().back(this.delta);
        }
        //diff l3
        Tensor_OP().mul(this.getLinear2().diff, active.getOutput(), tmp);
        getLinear3().back(tmp);
        //diff l1
        Tensor_OP().mul(this.getLinear2().diff, getLinear3().getOutput(), tmp);
        active.back(tmp);
        getLinear1().back(active.diff);
        Tensor_OP().add(this.getLinear1().diff, this.getLinear3().diff, this.getLinear1().diff);
        this.diff = this.getLinear1().diff;
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
        getLinear1().update();
        getLinear2().update();
        getLinear3().update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.feed_forward;
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
        getLinear3().saveModel(outputStream);
        getLinear2().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getLinear1().loadModel(inputStream);
        getLinear3().loadModel(inputStream);
        getLinear2().loadModel(inputStream);
    }

    public void putParamters() {
        getLinear1().putParamters();
        getLinear3().putParamters();
        getLinear2().putParamters();
    }

    public void putParamterGrads() {
        getLinear1().putParamterGrads();
        getLinear3().putParamterGrads();
        getLinear2().putParamterGrads();
    }

    public FullyLayer getLinear1() {
        return linear1;
    }

    public void setLinear1(FullyLayer linear1) {
        this.linear1 = linear1;
    }

    public FullyLayer getLinear2() {
        return linear2;
    }

    public void setLinear2(FullyLayer linear2) {
        this.linear2 = linear2;
    }

    public FullyLayer getLinear3() {
        return linear3;
    }

    public void setLinear3(FullyLayer linear3) {
        this.linear3 = linear3;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
        linear3.accGrad(scale);
    }
}

