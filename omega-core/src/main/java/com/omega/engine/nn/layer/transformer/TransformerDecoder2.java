package com.omega.engine.nn.layer.transformer;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.EmbeddingLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer Decoder Layer
 *
 * @author Administrator
 */
public class TransformerDecoder2 extends Layer {
    private int time;
    private int vocab_size;
    private int embedDim = 0;
    private boolean bias = false;
    private boolean dropout = false;
    private int headNum = 8;
    private int n_layers = 6;
    private EmbeddingLayer src_emb;
    private EmbeddingLayer pos_emb;
    private List<TransformerBlock> decoderLayers;
    //	private LNLayer ln;
    private Tensor positions;

    public TransformerDecoder2(int vocab_size, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout) {
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.vocab_size = vocab_size;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.dropout = dropout;
        this.initLayers();
    }

    public TransformerDecoder2(int vocab_size, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout, Network network) {
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.vocab_size = vocab_size;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.dropout = dropout;
        this.initLayers();
    }

    public void initLayers() {
        this.src_emb = new EmbeddingLayer(vocab_size, embedDim, network);
        this.src_emb.weight = new Tensor(1, 1, src_emb.width, src_emb.oWidth, RandomUtils.uniform(this.src_emb.width * this.src_emb.oWidth, 0.0f, 0.02f), true);
        this.pos_emb = new EmbeddingLayer(time, embedDim, network);
        this.pos_emb.weight = new Tensor(1, 1, pos_emb.width, pos_emb.oWidth, RandomUtils.uniform(this.pos_emb.width * this.pos_emb.oWidth, 0.0f, 0.02f), true);
        decoderLayers = new ArrayList<TransformerBlock>();
        for (int i = 0; i < n_layers; i++) {
            TransformerBlock decoderLayer = new TransformerBlock(headNum, time, embedDim, bias, dropout, network);
            decoderLayers.add(decoderLayer);
        }
        //		this.ln = new LNLayer(decoderLayers.get(decoderLayers.size() - 1));
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.getShape()[0];
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
        src_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
        Tensor decoderOutput = src_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).forward(decoderOutput);
            decoderOutput = decoderLayers.get(i).getOutput();
        }
        //		this.ln.forward(decoderOutput);
        //		this.output = this.ln.getOutput();
        this.output = decoderOutput;
    }

    public void output(Tensor positions) {
        // TODO Auto-generated method stub
        src_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
        Tensor decoderOutput = src_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).forward(decoderOutput);
            decoderOutput = decoderLayers.get(i).getOutput();
        }
        //		this.ln.forward(decoderOutput);
        //		this.output = this.ln.getOutput();
        this.output = decoderOutput;
    }

    public void output(Tensor mask, Tensor positions) {
        // TODO Auto-generated method stub
        src_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
        Tensor decoderOutput = src_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).forward(decoderOutput);
            decoderOutput = decoderLayers.get(i).getOutput();
        }
        //		this.ln.forward(decoderOutput);
        //		this.output = this.ln.getOutput();
        this.output = decoderOutput;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		this.ln.back(delta);
        //		Tensor decoderDiff = this.ln.diff;
        Tensor decoderDiff = delta;
        for (int i = n_layers - 1; i >= 0; i--) {
            decoderLayers.get(i).back(decoderDiff);
            decoderDiff = decoderLayers.get(i).diff;
        }
        src_emb.back(decoderDiff);
        pos_emb.back(decoderDiff);
        this.diff = this.src_emb.diff;
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

    public void forward(Tensor input, Tensor positions) {
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
        this.output(positions);
    }

    public void forward(Tensor input, Tensor mask, Tensor positions) {
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
        this.output(mask, positions);
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
        src_emb.update();
        pos_emb.update();
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.transformer_decoder;
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

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        src_emb.accGrad(scale);
        pos_emb.accGrad(scale);
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).accGrad(scale);
        }
    }
}

