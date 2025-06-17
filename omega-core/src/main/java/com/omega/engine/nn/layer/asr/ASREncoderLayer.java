package com.omega.engine.nn.layer.asr;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * ASREncoderLayer
 *
 * @author Administrator
 */
public class ASREncoderLayer extends Layer {
    public LNLayer ln1;
    public MultiHeadAttentionMaskLayer attn;
    public LNLayer ln2;
    /**
     * Position-wise Feedforward
     */
    public ASRPoswiseFeedForwardLinearLayer pos_ffn;
    private int time;
    private int headNum = 8;
    private int embedDim = 0;
    private int nChannel = 0;
    private boolean bias = false;
    private boolean dropout = false;
    private Tensor tmp1;
    private Tensor tmp2;

    public ASREncoderLayer(int headNum, int time, int embedDim, int nChannel, boolean bias, boolean dropout) {
        this.headNum = headNum;
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.dropout = dropout;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public ASREncoderLayer(int headNum, int time, int embedDim, int nChannel, boolean bias, boolean dropout, Network network) {
        this.headNum = headNum;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.dropout = dropout;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public void initLayers() {
        this.ln1 = new LNLayer(this, bias);
        this.attn = new MultiHeadAttentionMaskLayer(embedDim, embedDim, headNum, time, time, bias, dropout, network);
        this.ln2 = new LNLayer(attn, bias);
        this.pos_ffn = new ASRPoswiseFeedForwardLinearLayer(embedDim, nChannel, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.getShape()[0];
        if (this.tmp1 == null || this.tmp1.getShape()[0] != this.number) {
            this.tmp1 = Tensor.createTensor(this.tmp1, number, 1, 1, embedDim, true);
            this.tmp2 = Tensor.createTensor(this.tmp2, number, 1, 1, embedDim, true);
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
        ln1.forward(input);
        attn.forward(ln1.getOutput(), ln1.getOutput(), ln1.getOutput());
        Tensor_OP().add(attn.getOutput(), input, tmp1);
        ln2.forward(tmp1);
        pos_ffn.forward(ln2.getOutput());
        Tensor_OP().add(pos_ffn.getOutput(), tmp1, tmp2);
        this.output = tmp2;
    }

    public void output(Tensor mask) {
        // TODO Auto-generated method stub
        ln1.forward(input);
        attn.forward(ln1.getOutput(), ln1.getOutput(), ln1.getOutput(), mask);
        Tensor_OP().add(attn.getOutput(), input, tmp1);
        ln2.forward(tmp1);
        pos_ffn.forward(ln2.getOutput());
        Tensor_OP().add(pos_ffn.getOutput(), tmp1, tmp2);
        this.output = tmp2;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        pos_ffn.back(delta);
        ln2.back(pos_ffn.diff);
        Tensor_OP().add(ln2.diff, delta, ln2.diff);
        attn.back(ln2.diff);
        ln1.back(attn.diff);
        Tensor_OP().add(ln1.diff, ln2.diff, tmp2);
        this.diff = tmp2;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         *
         */
        this.setInput();
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         *
         */
        this.setDelta();
        /**
         * 计算梯度
         *
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
         *
         */
        this.setInput(input);
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output();
    }

    public void forward(Tensor input, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         *
         */
        this.setInput(input);
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output(mask);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         *
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         *
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        ln1.update();
        attn.update();
        ln2.update();
        pos_ffn.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.transformer_encoder;
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
        ln1.saveModel(outputStream);
        attn.saveModel(outputStream);
        ln2.saveModel(outputStream);
        pos_ffn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        ln1.loadModel(inputStream);
        attn.loadModel(inputStream);
        ln2.loadModel(inputStream);
        pos_ffn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        ln1.accGrad(scale);
        attn.accGrad(scale);
        ln2.accGrad(scale);
        pos_ffn.accGrad(scale);
    }
}
