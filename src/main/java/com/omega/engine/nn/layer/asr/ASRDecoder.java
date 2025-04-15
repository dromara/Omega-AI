package com.omega.engine.nn.layer.asr;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * ASRDecoder
 *
 * @author Administrator
 */
public class ASRDecoder extends Layer {
    public EmbeddingIDLayer tgt_emb;
    public EmbeddingIDLayer pos_emb;
    public List<ASRDecoderLayer> decoders;
    public RMSLayer norm;
    private int vocSize;
    private int time;
    private int kvTime;
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private int headNum = 8;
    private int n_layers = 4;
    private Tensor encoderDiff;

    public ASRDecoder(int vocSize, int time, int kvTime, int embedDim, int headNum, int nChannel, int n_layers, boolean bias) {
        this.vocSize = vocSize;
        this.time = time;
        this.kvTime = kvTime;
        this.headNum = headNum;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.bias = bias;
        this.initLayers();
    }

    public ASRDecoder(int vocSize, int time, int kvTime, int embedDim, int headNum, int nChannel, int n_layers, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.vocSize = vocSize;
        this.time = time;
        this.kvTime = kvTime;
        this.headNum = headNum;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.bias = bias;
        this.initLayers();
    }

    public Tensor pos_sinusoid_embedding(int time, int embedDim) {
        float[] data = new float[time * embedDim];
        for (int i = 0; i < time; i++) {
            for (int j = 0; j < embedDim; j++) {
                int idx = j / 2;
                double v = i / Math.pow(1e4, 2.0f * (idx * 1.0f / embedDim));
                if (j % 2 == 0) {
                    data[i * embedDim + j] = (float) Math.sin(v);
                } else {
                    data[i * embedDim + j] = (float) Math.cos(v);
                }
            }
        }
        return new Tensor(1, 1, time, embedDim, data, true);
    }

    public void initLayers() {
        this.tgt_emb = new EmbeddingIDLayer(vocSize, embedDim, network);
        this.pos_emb = new EmbeddingIDLayer(time, embedDim, true, network);
        pos_emb.weight = pos_sinusoid_embedding(time, embedDim);
        decoders = new ArrayList<ASRDecoderLayer>();
        for (int i = 0; i < n_layers; i++) {
            ASRDecoderLayer decoderLayer = new ASRDecoderLayer(headNum, time, embedDim, kvTime, nChannel, bias, false, network);
            decoders.add(decoderLayer);
        }
        this.norm = new RMSLayer(decoders.get(n_layers - 1));
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number / time;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (encoderDiff == null) {
            encoderDiff = Tensor.createGPUTensor(this.encoderDiff, number, kvTime, 1, embedDim, true);
        } else {
            encoderDiff.clearGPU();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    }

    public void output(Tensor enc_out, Tensor mask, Tensor en_de_mask, Tensor positions) {
        // TODO Auto-generated method stub
        tgt_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(tgt_emb.getOutput(), pos_emb.getOutput(), tgt_emb.getOutput());
        Tensor decoderOutput = tgt_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            decoders.get(i).forward(decoderOutput, enc_out, mask, en_de_mask);
            decoderOutput = decoders.get(i).getOutput();
        }
        norm.forward(decoderOutput);
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
        norm.back(delta);
        Tensor decoderDiff = norm.diff;
        for (int i = n_layers - 1; i >= 0; i--) {
            decoders.get(i).back(decoderDiff, encoderDiff);
            decoderDiff = decoders.get(i).diff;
        }
        tgt_emb.back(decoderDiff);
        // encoderDiff.showDM("encoderDiff");
        this.diff = encoderDiff;
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

    public void forward(Tensor input, Tensor enc_out, Tensor mask, Tensor en_de_mask, Tensor positions) {
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
        this.output(enc_out, mask, en_de_mask, positions);
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
        tgt_emb.update();
        for (int i = 0; i < n_layers; i++) {
            decoders.get(i).update();
        }
        norm.update();
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
        tgt_emb.accGrad(scale);
        for (int i = 0; i < n_layers; i++) {
            decoders.get(i).accGrad(scale);
        }
        norm.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        tgt_emb.saveModel(outputStream);
        for (int i = 0; i < n_layers; i++) {
            decoders.get(i).saveModel(outputStream);
        }
        norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        tgt_emb.loadModel(inputStream);
        for (int i = 0; i < n_layers; i++) {
            decoders.get(i).loadModel(inputStream);
        }
        norm.loadModel(inputStream);
    }
}
