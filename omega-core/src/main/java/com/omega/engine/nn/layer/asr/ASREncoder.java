package com.omega.engine.nn.layer.asr;

import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * ASREncoder
 *
 * @author Administrator
 */
public class ASREncoder extends Layer {
    public FullyLayer feature_emb;
    public EmbeddingIDLayer pos_emb;
    public List<ASREncoderLayer> encoders;
    private int wavDim;
    private int time;
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private int headNum = 8;
    private int n_layers = 4;
    private Tensor positions;

    public ASREncoder(int wavDim, int time, int embedDim, int headNum, int nChannel, int n_layers, boolean bias) {
        this.wavDim = wavDim;
        this.time = time;
        this.headNum = headNum;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.bias = bias;
        this.initLayers();
    }

    public ASREncoder(int wavDim, int time, int embedDim, int headNum, int nChannel, int n_layers, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.wavDim = wavDim;
        this.time = time;
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
        this.feature_emb = new FullyLayer(wavDim, embedDim, bias, network);
        this.pos_emb = new EmbeddingIDLayer(time, embedDim, true, network);
        pos_emb.weight = pos_sinusoid_embedding(time, embedDim);
        encoders = new ArrayList<ASREncoderLayer>();
        for (int i = 0; i < n_layers; i++) {
            ASREncoderLayer decoderLayer = new ASREncoderLayer(headNum, time, embedDim, nChannel, bias, false, network);
            encoders.add(decoderLayer);
        }
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
        feature_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(feature_emb.getOutput(), pos_emb.getOutput(), feature_emb.getOutput());
        Tensor encoderOutput = feature_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).forward(encoderOutput);
            encoderOutput = encoders.get(i).getOutput();
        }
        this.output = encoderOutput;
    }

    public void output(Tensor mask, Tensor positions) {
        // TODO Auto-generated method stub
        feature_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(feature_emb.getOutput(), pos_emb.getOutput(), feature_emb.getOutput());
        Tensor encoderOutput = feature_emb.getOutput();
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).forward(encoderOutput, mask);
            encoderOutput = encoders.get(i).getOutput();
        }
        // encoderOutput.showDM();
        this.output = encoderOutput;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        Tensor encoderDiff = delta;
        for (int i = n_layers - 1; i >= 0; i--) {
            encoders.get(i).back(encoderDiff);
            encoderDiff = encoders.get(i).diff;
        }
        // encoderDiff.showDM("encoderDiff");
        feature_emb.back(encoderDiff);
        this.diff = feature_emb.diff;
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

    public void forward(Tensor input, Tensor mask, Tensor positions) {
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
        this.output(mask, positions);
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
        feature_emb.update();
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).update();
        }
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

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        feature_emb.accGrad(scale);
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).accGrad(scale);
        }
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        feature_emb.saveModel(outputStream);
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        feature_emb.loadModel(inputStream);
        for (int i = 0; i < n_layers; i++) {
            encoders.get(i).loadModel(inputStream);
        }
    }
}
