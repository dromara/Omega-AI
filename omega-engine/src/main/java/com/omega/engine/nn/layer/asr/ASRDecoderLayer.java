package com.omega.engine.nn.layer.asr;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * ASRDecoderLayer
 *
 * @author Administrator
 */
public class ASRDecoderLayer extends Layer {
    public LNLayer ln1;
    public MultiHeadAttentionMaskLayer attn;
    public LNLayer ln2;
    public MultiHeadAttentionMaskLayer cross_attn;
    public LNLayer ln3;
    /**
     * Position-wise Feedforward
     */
    public ASRPoswiseFeedForwardLinearLayer pos_ffn;
    private int time;
    private int kvTime;
    private int headNum = 8;
    private int embedDim = 0;
    private int nChannel = 0;
    private boolean bias = false;
    private boolean dropout = false;
    private Tensor tmp1;
    private Tensor tmp2;
    private Tensor tmp3;

    public ASRDecoderLayer(int headNum, int time, int embedDim, int kvTime, int nChannel, boolean bias, boolean dropout) {
        this.headNum = headNum;
        this.time = time;
        this.kvTime = kvTime;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.dropout = dropout;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public ASRDecoderLayer(int headNum, int time, int embedDim, int kvTime, int nChannel, boolean bias, boolean dropout, Network network) {
        this.headNum = headNum;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.kvTime = kvTime;
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
        this.cross_attn = new MultiHeadAttentionMaskLayer(embedDim, embedDim, headNum, time, kvTime, bias, dropout, network);
        this.ln3 = new LNLayer(cross_attn, bias);
        this.pos_ffn = new ASRPoswiseFeedForwardLinearLayer(embedDim, nChannel, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if (this.tmp1 == null || this.tmp1.number != this.number) {
            this.tmp1 = Tensor.createTensor(this.tmp1, number, 1, 1, embedDim, true);
            this.tmp2 = Tensor.createTensor(this.tmp2, number, 1, 1, embedDim, true);
            this.tmp3 = Tensor.createTensor(this.tmp3, number, 1, 1, embedDim, true);
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
    }

    /**
     * @param den_out
     * @param decode_mask must be triu mask
     * @param en_de_mask  [decode_dim, encoder_dim]
     */
    public void output(Tensor en_out, Tensor decode_mask, Tensor en_de_mask) {
        // TODO Auto-generated method stub
        ln1.forward(input);
        attn.forward(ln1.getOutput(), ln1.getOutput(), ln1.getOutput(), decode_mask);
        Tensor_OP().add(attn.getOutput(), input, tmp1);
        ln2.forward(tmp1);
        cross_attn.forward(ln2.getOutput(), en_out, en_out, en_de_mask);
        Tensor_OP().add(cross_attn.getOutput(), tmp1, tmp2);
        //		tmp2.showDM("dec_out2");
        ln3.forward(tmp2);
        pos_ffn.forward(ln3.getOutput());
        Tensor_OP().add(pos_ffn.getOutput(), tmp2, tmp3);
        this.output = tmp3;
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

    public void diff(Tensor encodeDiff) {
        // TODO Auto-generated method stub
        pos_ffn.back(delta);
        ln3.back(pos_ffn.diff);
        Tensor_OP().add(ln3.diff, delta, ln3.diff);
        cross_attn.back(ln3.diff, encodeDiff);
        ln2.back(cross_attn.diff);
        Tensor_OP().add(ln2.diff, ln3.diff, ln2.diff);
        //		ln2.diff.showDM("ln2.diff");
        attn.back(ln2.diff);
        //		attn.diff.showDM("attn.diff");
        ln1.back(attn.diff);
        Tensor_OP().add(ln1.diff, ln2.diff, ln1.diff);
        this.diff = ln1.diff;
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

    public void forward(Tensor input, Tensor en_out, Tensor decode_mask, Tensor en_de_mask) {
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
        this.output(en_out, decode_mask, en_de_mask);
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

    public void back(Tensor delta, Tensor encodeDiff) {
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
        this.diff(encodeDiff);
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
        cross_attn.update();
        ln3.update();
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
        cross_attn.saveModel(outputStream);
        ln3.saveModel(outputStream);
        pos_ffn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        ln1.loadModel(inputStream);
        attn.loadModel(inputStream);
        ln2.loadModel(inputStream);
        cross_attn.loadModel(inputStream);
        ln3.loadModel(inputStream);
        pos_ffn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        ln1.accGrad(scale);
        attn.accGrad(scale);
        ln2.accGrad(scale);
        cross_attn.accGrad(scale);
        ln3.accGrad(scale);
        pos_ffn.accGrad(scale);
    }
}
