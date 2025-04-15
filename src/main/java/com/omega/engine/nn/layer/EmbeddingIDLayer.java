package com.omega.engine.nn.layer;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.gpu.EmbeddingKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * FullyLayer
 *
 * @author Administrator
 */
public class EmbeddingIDLayer extends Layer {
    private EmbeddingKernel kernel;
    private Tensor factor;

    public EmbeddingIDLayer(int num_embeddings, int embedding_dim) {
        this.channel = 1;
        this.height = 1;
        this.width = num_embeddings;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = embedding_dim;
        this.hasParams = true;
        this.hasBias = false;
        this.initParam();
    }

    public EmbeddingIDLayer(int num_embeddings, int embedding_dim, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = 1;
        this.height = 1;
        this.width = num_embeddings;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = embedding_dim;
        this.hasParams = true;
        this.hasBias = false;
        network.paramLayers.add(this);
        this.initParam();
    }

    public EmbeddingIDLayer(int num_embeddings, int embedding_dim, boolean freeze, Network network) {
        this.network = network;
        this.freeze = freeze;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = 1;
        this.height = 1;
        this.width = num_embeddings;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = embedding_dim;
        this.hasParams = true;
        this.hasBias = false;
        network.paramLayers.add(this);
        this.initParam();
    }

    public static float[] outer(float[] a, float[] b) {
        float[] o = new float[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                o[i * b.length + j] = a[i] * b[j];
            }
        }
        return o;
    }

    public static float[] stack(float[] a, float[] b) {
        float[] o = new float[a.length + b.length];
        for (int i = 0; i < a.length; i++) {
            o[i * 2] = a[i];
            o[i * 2 + 1] = b[i];
        }
        return o;
    }

    public static float[] cat(float[] a, float[] b) {
        float[] o = new float[a.length + b.length];
        for (int i = 0; i < a.length; i++) {
            o[i] = a[i];
            o[a.length + i] = b[i];
        }
        return o;
    }

    public static void main(String[] args) {
        int T = 1000;
        int d_model = 128;
        float[] emb = MatrixUtils.order(d_model / 2, 0, (float) (-2.0f / d_model * Math.log(10000)));
        emb = MatrixOperation.exp(emb);
        //		System.out.println(JsonUtils.toJson(emb));
        float[] pos = MatrixUtils.order(T, 0, 1);
        float[] o = outer(pos, emb);
        float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        //		System.out.println(JsonUtils.toJson(o));
        //		System.out.println(JsonUtils.toJson(cos));
        //		System.out.println(JsonUtils.toJson(sin));
        float[] wd = stack(sin, cos);
        Tensor weight = new Tensor(1, 1, T, d_model, wd, true);
        PrintUtils.printImage(weight);
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.number != this.diff.number) {
            this.diff = new Tensor(number, channel, height, width, true, true);
            this.diffW = new Tensor(1, 1, width, oWidth, true, true);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        if (kernel == null) {
            kernel = new EmbeddingKernel(cuda());
        }
        this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaiming_uniform(this.width * this.oWidth, this.width, this.paramsInit), true);
        //		this.weight = new Tensor(1, 1, width, oWidth, MatrixUtils.order(this.width * this.oWidth, 0.001f, 0.001f), true);
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		if(this.input != null) {
        kernel.forward(input, this.weight, output);
        //		}
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        diffW.clearGPU();
        kernel.backward(delta, diffW, input);
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
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

    /**
     * w(t) = w(t-1) + θ * deltaW
     * <p>
     * b(t) = b(t-1) + θ * deltaB
     * <p>
     * θ : learningRate
     */
    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffW);
            }
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.data[i] -= this.learnRate * this.diffW.data[i];
                }
            }
            this.clearAccGrad();
        }
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (accDW == null) {
            accDW = diffW.copyGPU();
        } else {
            kernel.axpy_gpu(diffW, accDW, accDW.dataLength, scale, 1, 1);
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.embedding;
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void getTimeEmbedding(Tensor input) {
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        kernel.get_time_embedding(input, factor, this.output, oWidth / 2);
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

    public void clear() {
        //		this.output.clear();
        //		this.diffW.clear();
        //		this.diff.clear();
        //		this.diffW.clearGPU();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        ModelUtils.saveParams(outputStream, weight);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        ModelUtils.loadParams(inputStream, weight);
    }

    public void putParamters() {
        this.network.addPamamter(weight);
    }

    public void putParamterGrads() {
        this.network.addDeltaParamters(diffW);
    }

    public Tensor createTimeEMB(int T, int d_model) {
        float[] emb = MatrixUtils.order(d_model / 2, 0, (float) (-2.0f / d_model * Math.log(10000)));
        emb = MatrixOperation.exp(emb);
        float[] pos = MatrixUtils.order(T, 0, 1);
        float[] o = outer(pos, emb);
        float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        float[] wd = stack(sin, cos);
        Tensor weight = new Tensor(1, 1, T, d_model, wd, true);
        return weight;
    }

    public void initFactor(int T, int d_model) {
        float[] emb = MatrixUtils.order(d_model / 2, 0, 1.0f / (d_model / 2));
        //		System.err.println(JsonUtils.toJson(emb));
        for (int i = 0; i < emb.length; i++) {
            emb[i] = (float) Math.pow(10000, emb[i]);
        }
        factor = new Tensor(emb.length, 1, 1, 1, emb, true);
    }

    public Tensor getTimeEMB(int T, int d_model) {
        float[] emb = MatrixUtils.order(d_model / 2, 0, 1.0f / (d_model / 2));
        //		System.err.println(JsonUtils.toJson(emb));
        for (int i = 0; i < emb.length; i++) {
            emb[i] = (float) Math.pow(10000, emb[i]);
        }
        //		System.err.println(JsonUtils.toJson(emb));
        float[] pos = MatrixUtils.order(T, 0, 1);
        float[] o = outer(pos, emb);
        float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        float[] wd = cat(sin, cos);
        Tensor weight = new Tensor(1, 1, T, d_model, wd, true);
        return weight;
    }
}

