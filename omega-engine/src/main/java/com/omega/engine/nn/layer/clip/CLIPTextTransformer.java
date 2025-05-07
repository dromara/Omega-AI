package com.omega.engine.nn.layer.clip;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * CLIPVisionTransformer
 *
 * @author Administrator
 */
public class CLIPTextTransformer extends Layer {
    private int vocabSize = 49408;
    private int intermediateSize = 2048;
    private int maxPositionEmbeddings = 77;
    private int time;
    private int embedDim = 512;
    private boolean bias = false;
    private int headNum = 8;
    private int n_layers = 12;
    private CLIPTextEmbeddingLayer embeddings;
    private List<CLIPEncoderLayer> encoders;
    private LNLayer finalLayerNorm;
    private Tensor imageEncoders;

    public CLIPTextTransformer(int vocabSize, int maxPositionEmbeddings, int n_layers, int headNum, int time, int embedDim, boolean bias) {
        this.vocabSize = vocabSize;
        this.time = time;
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public CLIPTextTransformer(int vocabSize, int maxPositionEmbeddings, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout, Network network) {
        this.vocabSize = vocabSize;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public void initLayers() {
        embeddings = new CLIPTextEmbeddingLayer(vocabSize, embedDim, maxPositionEmbeddings, network);
        encoders = new ArrayList<CLIPEncoderLayer>();
        for (int i = 0; i < n_layers; i++) {
            CLIPEncoderLayer encoder = new CLIPEncoderLayer(headNum, time, embedDim, intermediateSize, bias, false, true, network);
            getEncoders().add(encoder);
        }
        finalLayerNorm = new LNLayer(getEncoders().get(n_layers - 1));
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
        getEmbeddings().forward(input);
        Tensor out1 = getEmbeddings().getOutput().view(getEmbeddings().getOutput().number * getEmbeddings().getOutput().channel, 1, 1, getEmbeddings().getOutput().width);
        for (int i = 0; i < n_layers; i++) {
            getEncoders().get(i).forward(out1);
            out1 = getEncoders().get(i).getOutput();
        }
        getFinalLayerNorm().forward(out1);
        getEmbeddings().getOutput().viewOrg();
        this.output = getFinalLayerNorm().getOutput();
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
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public List<CLIPEncoderLayer> getEncoders() {
        return encoders;
    }

    public Tensor getImageEncoders() {
        return imageEncoders;
    }

    public CLIPTextEmbeddingLayer getEmbeddings() {
        return embeddings;
    }

    public LNLayer getFinalLayerNorm() {
        return finalLayerNorm;
    }
}

