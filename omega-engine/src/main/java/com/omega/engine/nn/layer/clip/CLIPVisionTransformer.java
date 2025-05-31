package com.omega.engine.nn.layer.clip;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
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
public class CLIPVisionTransformer extends Layer {
    private int channel = 3;
    private int imgSize = 224;
    private int patchSize = 32;
    private int intermediateSize = 3072;
    private int time;
    private int embedDim = 0;
    private boolean bias = false;
    private int headNum = 8;
    private int n_layers = 6;
    private CLIPVisionEmbeddingLayer embeddings;
    private LNLayer preLayrnorm;
    private List<CLIPEncoderLayer> encoders;
    private LNLayer postLayernorm;
    private Tensor imageEncoders;

    public CLIPVisionTransformer(int channel, int imgSize, int patchSize, int n_layers, int headNum, int time, int embedDim, boolean bias) {
        this.imgSize = imgSize;
        this.channel = channel;
        this.patchSize = patchSize;
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

    public CLIPVisionTransformer(int channel, int imgSize, int patchSize, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout, Network network) {
        this.channel = channel;
        this.imgSize = imgSize;
        this.patchSize = patchSize;
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
        embeddings = new CLIPVisionEmbeddingLayer(channel, imgSize, imgSize, embedDim, imgSize, patchSize, bias, network);
        preLayrnorm = new LNLayer(getEmbeddings());
        encoders = new ArrayList<CLIPEncoderLayer>();
        for (int i = 0; i < n_layers; i++) {
            CLIPEncoderLayer encoder = new CLIPEncoderLayer(headNum, time, embedDim, intermediateSize, bias, false, false, network);
            getEncoders().add(encoder);
        }
        postLayernorm = new LNLayer(getEncoders().get(n_layers - 1));
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
        Tensor emb = getEmbeddings().getOutput().view(getEmbeddings().getOutput().number * getEmbeddings().getOutput().channel, 1, 1, getEmbeddings().getOutput().width);
        getPreLayrnorm().forward(emb);
        Tensor out1 = getPreLayrnorm().getOutput();
        for (int i = 0; i < n_layers; i++) {
            getEncoders().get(i).forward(out1);
            out1 = getEncoders().get(i).getOutput();
        }
        imageEncoders = out1;
        getPostLayernorm().forward(out1);
        getEmbeddings().getOutput().viewOrg();
        this.output = getPostLayernorm().getOutput();
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
        //
        //		getSrc_emb().saveModel(outputStream);
        //
        //		for(int i = 0;i<n_layers;i++) {
        //			getDecoderLayers().get(i).saveModel(outputStream);
        //		}
        //
        //		getNorm().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        //
        //		getSrc_emb().loadModel(inputStream);
        //
        //		for(int i = 0;i<n_layers;i++) {
        //			getDecoderLayers().get(i).loadModel(inputStream);
        //		}
        //
        //		getNorm().loadModel(inputStream);
        //
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public CLIPVisionEmbeddingLayer getEmbeddings() {
        return embeddings;
    }

    public LNLayer getPreLayrnorm() {
        return preLayrnorm;
    }

    public List<CLIPEncoderLayer> getEncoders() {
        return encoders;
    }

    public LNLayer getPostLayernorm() {
        return postLayernorm;
    }

    public Tensor getImageEncoders() {
        return imageEncoders;
    }
}

