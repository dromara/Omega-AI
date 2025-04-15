package com.omega.engine.nn.layer.clip.bert;

import com.omega.common.tensor.Tensor;
import com.omega.utils.RandomUtils;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * BertEmbeddings Layer
 *
 * @author Administrator
 */
public class BertEmbeddingsLayer extends Layer {
    public EmbeddingIDLayer wordEmbeddings;
    public EmbeddingIDLayer positionEmbeddings;
    public EmbeddingIDLayer tokenTypeEmbeddings;
    public LNLayer norm;
    private int vocabSize;
    private int hiddenSize;
    private int maxPositionEmbeddingsSize;
    private int typeVocabSize;
    private Tensor positionIDS;

    public BertEmbeddingsLayer(int vocabSize, int hiddenSize, int maxPositionEmbeddingsSize, int typeVocabSize) {
        this.vocabSize = vocabSize;
        this.typeVocabSize = typeVocabSize;
        this.maxPositionEmbeddingsSize = maxPositionEmbeddingsSize;
        this.hiddenSize = hiddenSize;
        this.channel = 1;
        this.height = 1;
        this.width = hiddenSize;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public BertEmbeddingsLayer(int vocabSize, int hiddenSize, int maxPositionEmbeddingsSize, int typeVocabSize, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.vocabSize = vocabSize;
        this.typeVocabSize = typeVocabSize;
        this.maxPositionEmbeddingsSize = maxPositionEmbeddingsSize;
        this.hiddenSize = hiddenSize;
        this.channel = 1;
        this.height = 1;
        this.width = hiddenSize;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.wordEmbeddings = new EmbeddingIDLayer(vocabSize, hiddenSize, network);
        this.positionEmbeddings = new EmbeddingIDLayer(maxPositionEmbeddingsSize, hiddenSize, network);
        this.tokenTypeEmbeddings = new EmbeddingIDLayer(typeVocabSize, hiddenSize, network);
        this.norm = new LNLayer(wordEmbeddings);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if (positionIDS == null || positionIDS.number != this.number * maxPositionEmbeddingsSize) {
            float[] data = RandomUtils.orderAndUnsqueeze(maxPositionEmbeddingsSize, this.number, 1.0f, 0.0f);
            positionIDS = new Tensor(this.number * maxPositionEmbeddingsSize, 1, 1, 1, data, true);
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

    public void output(Tensor tokenIds) {
        // TODO Auto-generated method stub
        wordEmbeddings.forward(input);
        positionEmbeddings.forward(positionIDS);
        tokenTypeEmbeddings.forward(tokenIds);
        Tensor_OP().add(wordEmbeddings.getOutput(), positionEmbeddings.getOutput(), wordEmbeddings.getOutput());
        Tensor_OP().add(wordEmbeddings.getOutput(), tokenTypeEmbeddings.getOutput(), wordEmbeddings.getOutput());
        norm.forward(wordEmbeddings.getOutput());
        this.output = norm.getOutput();
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

    public void forward(Tensor input, Tensor tokenIds) {
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
        this.output(tokenIds);
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
        return LayerType.bert_output;
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
}

