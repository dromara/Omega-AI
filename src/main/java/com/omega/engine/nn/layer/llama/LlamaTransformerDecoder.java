package com.omega.engine.nn.layer.llama;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * Transformer Decoder Layer
 *
 * @author Administrator
 */
public class LlamaTransformerDecoder extends Layer {
    private int time;
    private int vocab_size;
    private int embedDim = 0;
    private boolean flashAttention = false;
    private boolean bias = false;
    private boolean dropout = false;
    private int multiple_of = 64;
    private int headNum = 8;
    private int nKVHeadNum = 8;
    private int n_layers = 6;
    private EmbeddingIDLayer src_emb;
    private List<LlamaTransformerBlock> decoderLayers;
    private RMSLayer norm;
    private DropoutLayer dropoutLayer;

    public LlamaTransformerDecoder(int vocab_size, int n_layers, int headNum, int time, int embedDim, int multiple_of, boolean bias, boolean dropout) {
        this.headNum = headNum;
        this.nKVHeadNum = headNum;
        this.multiple_of = multiple_of;
        this.n_layers = n_layers;
        this.vocab_size = vocab_size;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.dropout = dropout;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public LlamaTransformerDecoder(int vocab_size, int n_layers, int headNum, int time, int embedDim, int multiple_of, boolean bias, boolean dropout, boolean flashAttention, Network network) {
        this.flashAttention = flashAttention;
        this.headNum = headNum;
        this.nKVHeadNum = headNum;
        this.multiple_of = multiple_of;
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
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public LlamaTransformerDecoder(int vocab_size, int n_layers, int headNum, int nKVHeadNum, int time, int embedDim, int multiple_of, boolean bias, boolean dropout, boolean flashAttention, Network network) {
        this.flashAttention = flashAttention;
        this.headNum = headNum;
        this.nKVHeadNum = nKVHeadNum;
        this.multiple_of = multiple_of;
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
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public void initLayers() {
        this.setSrc_emb(new EmbeddingIDLayer(vocab_size, embedDim, network));
        this.getSrc_emb().weight = new Tensor(1, 1, getSrc_emb().width, getSrc_emb().oWidth, RandomUtils.gaussianRandom(this.src_emb.width * this.src_emb.oWidth, 0.0f, 0.02f), true);
        //		this.src_emb.weight = new Tensor(1, 1, src_emb.width, src_emb.oWidth, RandomUtils.order(this.src_emb.width * this.src_emb.oWidth, 0.001f, 0.001f), true);
        setDecoderLayers(new ArrayList<LlamaTransformerBlock>());
        for (int i = 0; i < n_layers; i++) {
            LlamaTransformerBlock decoderLayer = new LlamaTransformerBlock(headNum, nKVHeadNum, time, embedDim, multiple_of, bias, dropout, flashAttention, network);
            getDecoderLayers().add(decoderLayer);
        }
        this.setNorm(new RMSLayer(getDecoderLayers().get(n_layers - 1)));
        if (dropout) {
            dropoutLayer = new DropoutLayer(0.1f, getSrc_emb());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        this.time = this.network.time;
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

    public void output(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        //		input.showDMByOffset(0, 100);
        getSrc_emb().forward(input);
        Tensor out1 = getSrc_emb().getOutput();
        //		out1.showDMByOffset(0, 100);
        if (dropout) {
            this.dropoutLayer.forward(out1);
            out1 = dropoutLayer.getOutput();
        }
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).forward(cos, sin, out1);
            out1 = getDecoderLayers().get(i).getOutput();
        }
        //		out1.showDMByOffset(0, 100);
        this.getNorm().forward(out1);
        //		this.getNorm().getOutput().showDMByOffset(0, 100);;
        this.output = this.getNorm().getOutput();
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

    public void diff(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        this.getNorm().back(delta);
        Tensor decoderDiff = this.getNorm().diff;
        //		long start27 = System.nanoTime();
        //		System.err.println("---decoder-norm:");
        //		decoderDiff.showDMByOffset(0, 100);
        for (int i = n_layers - 1; i >= 0; i--) {
            getDecoderLayers().get(i).back(cos, sin, decoderDiff);
            decoderDiff = getDecoderLayers().get(i).diff;
        }
        //		System.out.println("decoders:"+(System.nanoTime() - start27) / 1e6+"ms.");
        //		decoderDiff.showDMByNumber(0);
        if (dropout) {
            this.dropoutLayer.back(decoderDiff);
            decoderDiff = dropoutLayer.diff;
        }
        //		System.err.println("---decoder:");
        //		decoderDiff.showDMByOffset(0, 100);
        //		System.err.println("decoderDiff:");
        //		decoderDiff.showDM();
        //		long start26 = System.nanoTime();
        getSrc_emb().back(decoderDiff);
        //		System.out.println("emb:"+(System.nanoTime() - start26) / 1e6+"ms.");
        this.diff = this.getSrc_emb().diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
    }

    public void forward(Tensor cos, Tensor sin, Tensor input) {
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
        this.output(cos, sin);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
    }

    public void back(Tensor cos, Tensor sin, Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(cos, sin);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        getSrc_emb().update();
        getNorm().update();
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).update();
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        getSrc_emb().saveModel(outputStream);
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).saveModel(outputStream);
        }
        getNorm().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getSrc_emb().loadModel(inputStream);
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).loadModel(inputStream);
        }
        getNorm().loadModel(inputStream);
    }

    public void putParamters() {
        getSrc_emb().putParamters();
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).putParamters();
        }
        getNorm().putParamters();
    }

    public void putParamterGrads() {
        getSrc_emb().putParamterGrads();
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).putParamterGrads();
        }
        getNorm().putParamterGrads();
    }

    public EmbeddingIDLayer getSrc_emb() {
        return src_emb;
    }

    public void setSrc_emb(EmbeddingIDLayer src_emb) {
        this.src_emb = src_emb;
    }

    public List<LlamaTransformerBlock> getDecoderLayers() {
        return decoderLayers;
    }

    public void setDecoderLayers(List<LlamaTransformerBlock> decoderLayers) {
        this.decoderLayers = decoderLayers;
    }

    public RMSLayer getNorm() {
        return norm;
    }

    public void setNorm(RMSLayer norm) {
        this.norm = norm;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        getSrc_emb().accGrad(scale);
        for (int i = 0; i < n_layers; i++) {
            getDecoderLayers().get(i).accGrad(scale);
        }
        getNorm().accGrad(scale);
    }
}

