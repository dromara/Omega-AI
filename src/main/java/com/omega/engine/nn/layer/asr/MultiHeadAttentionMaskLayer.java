package com.omega.engine.nn.layer.asr;

import com.omega.common.config.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

/**
 * MultiHeadAttentionMaskLayer
 *
 * @author Administrator
 */
public class MultiHeadAttentionMaskLayer extends Layer {
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    private int time;
    private int kvTime;
    private int headNum = 1;
    private int embedDim = 0;
    private int kvDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private DropoutLayer dropoutLayer;
    private DropoutLayer dropoutLayer2;
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor dqt;
    private Tensor dkt;
    private Tensor dvt;
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    private Tensor dattn;
    private Tensor mask;
    private float maskFill = -1e4f;
    private int batchSize = 1;
    private boolean dropout = false;

    public MultiHeadAttentionMaskLayer(int embedDim, int kvDim, int headNum, int time, int kvTime, boolean bias, boolean dropout) {
        this.bias = bias;
        this.time = time;
        this.kvTime = kvTime;
        this.embedDim = embedDim;
        this.kvDim = kvDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.dropout = dropout;
        this.initLayers();
    }

    public MultiHeadAttentionMaskLayer(int embedDim, int kvDim, int headNum, int time, int kvTime, boolean bias, boolean dropout, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.kvTime = kvTime;
        this.embedDim = embedDim;
        this.kvDim = kvDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.dropout = dropout;
        this.initLayers();
    }

    public static void main(String[] args) {
        int embedDim = 256;
        int headNum = 8;
        int batchSize = 2;
        int time = 256;
        int context_time = 64;
        int context_dim = 512;
        Transformer tf = new Transformer();
        tf.number = batchSize;
        tf.time = time;
        tf.updater = UpdaterType.adamw;
        tf.CUDNN = true;
        tf.learnRate = 0.001f;
        tf.RUN_MODEL = RunModel.TRAIN;
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.0001f, 0.0001f);
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        float[] cdata = RandomUtils.order(batchSize * context_time * context_dim, 0.001f, 0.001f);
        Tensor context = new Tensor(batchSize * context_time, 1, 1, context_dim, cdata, true);
        //		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        MultiHeadAttentionMaskLayer mal = new MultiHeadAttentionMaskLayer(embedDim, context_dim, headNum, time, context_time, false, false, tf);
        String weight = "H:\\model\\crossAttn.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), mal, true);
        for (int i = 0; i < 10; i++) {
            //			input.showDM();
            tf.train_time++;
            mal.forward(input, context, context);
            mal.getOutput().showShape();
            mal.getOutput().showDM("output");
            mal.back(delta);
            //			delta.showDM();
            mal.diff.showDM("diff");
            //			mal.update();
            //			delta.copyData(tmp);
        }
    }

    public static void loadWeight(Map<String, Object> weightMap, MultiHeadAttentionMaskLayer network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        ClipModelUtils.loadData(network.qLinerLayer.weight, weightMap, "q_proj.weight");
        ClipModelUtils.loadData(network.kLinerLayer.weight, weightMap, "k_proj.weight");
        ClipModelUtils.loadData(network.vLinerLayer.weight, weightMap, "v_proj.weight");
        ClipModelUtils.loadData(network.oLinerLayer.weight, weightMap, "out_proj.weight");
        ClipModelUtils.loadData(network.oLinerLayer.bias, weightMap, "out_proj.bias");
    }

    public static boolean same(Tensor a, Tensor b) {
        float[] ad = a.syncHost();
        float[] bd = b.syncHost();
        for (int i = 0; i < ad.length; i++) {
            if (ad[i] != bd[i]) {
                System.out.println(ad[i] + ":" + bd[i] + "[" + i + "]");
                return false;
            }
        }
        return true;
    }

    public void initLayers() {
        this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
        this.kLinerLayer = new FullyLayer(kvDim, embedDim, bias, this.network);
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, kvDim, RandomUtils.order(this.embedDim * this.kvDim, 0.01f, 0.01f), true);
        this.vLinerLayer = new FullyLayer(kvDim, embedDim, bias, this.network);
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, kvDim, RandomUtils.order(this.embedDim * this.kvDim, 0.01f, 0.01f), true);
        this.oLinerLayer = new FullyLayer(embedDim, embedDim, true, this.network);
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
        if (this.dropout) {
            this.dropoutLayer = new DropoutLayer(0.1f, this.network);
            this.dropoutLayer2 = new DropoutLayer(0.1f, getoLinerLayer());
        }
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(network.cudaManager);
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(kvTime, 1, 1, network.cudaManager);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number / time;
        if (this.qt != null) {
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
            this.oi.viewOrg();
            this.qLinerLayer.getOutput().viewOrg();
            this.kLinerLayer.getOutput().viewOrg();
            this.vLinerLayer.getOutput().viewOrg();
            this.oLinerLayer.getOutput().viewOrg();
        }
        if (this.qt == null || this.qt.number != this.batchSize) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, kvTime, dk, true);
            this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, kvTime, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            if (kvTime < dk) {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
            } else {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, kvTime, true);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, kvTime, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, kvTime, dk, true);
            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, kvTime, dk, true);
            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, kvTime, true);
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

    public void output(Tensor q, Tensor k, Tensor v) {
        // TODO Auto-generated method stub
        //		context.showShape();
        this.qLinerLayer.forward(q);
        this.kLinerLayer.forward(k);
        this.vLinerLayer.forward(v);
        Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.kLinerLayer.getOutput().view(batchSize, kvTime, headNum, dk);
        Tensor value = this.vLinerLayer.getOutput().view(batchSize, kvTime, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi);
        //		oLinerLayer.weight.showDM("olw");
        //		oLinerLayer.bias.showDM("olb");
        this.output = this.oLinerLayer.getOutput();
        //		output.showDMByOffsetRed(10 * output.height * output.width, output.height * output.width, "output----");
        if (dropout) {
            dropoutLayer2.forward(this.getoLinerLayer().getOutput());
            this.output = dropoutLayer2.getOutput();
        }
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, key.getGpuData(), dk, kvTime * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);
        if (mask != null) {
            Tensor_OP().bool(preatt, mask, preatt, maskFill);
        }
        softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        Tensor tmp = attn;
        if (dropout) {
            dropoutLayer.forward(attn);
            tmp = dropoutLayer.getOutput();
        }
        Tensor vaccum = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, kvTime, 1.0f, value.getGpuData(), dk, kvTime * dk, tmp.getGpuData(), kvTime, time * kvTime, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;
        if (dropout) {
            tmp = dropoutLayer.getOutput();
        }
        Tensor dvaccum = temp;
        /**
         * backward into dattn[b, nh, t, t2] vt[b, nh, t2, dk] -> [b, nh, dk, t2]
         * dvaccum[b, nh, t, dk]
         *
         */
        GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, vt.getGpuData(), dk, kvTime * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        /**
         * backward into dvt[b, nh, t2, dk] dvaccum[b, nh, t, dk] attn[b, nh, t, t2] ->
         * [b, nh, t2, t]
         *
         */
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, kvTime, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), kvTime, time * kvTime, 0.0f, dvt.getGpuData(), dk, kvTime * dk, batchSize * headNum);
        if (dropout) {
            dropoutLayer.back(dattn);
            dattn = dropoutLayer.diff;
        }
        // backward into preatt
        float d_k = (float) (1.0f / Math.sqrt(dk));
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        /**
         * backward into dqt dpreatt * kt
         *
         */
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, kvTime, 1.0f, kt.getGpuData(), dk, kvTime * dk, dpreatt.getGpuData(), kvTime, time * kvTime, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        /**
         * backward into dkt
         *
         */
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, kvTime, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), kvTime, time * kvTime, 0.0f, dkt.getGpuData(), dk, kvTime * dk, batchSize * headNum);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        if (dropout) {
            dropoutLayer2.back(delta);
            this.getoLinerLayer().back(dropoutLayer2.diff, oi);
        } else {
            this.getoLinerLayer().back(delta, oi);
        }
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.qLinerLayer.getOutput().shape());
        kt.view(this.kLinerLayer.getOutput().shape());
        vt.view(this.vLinerLayer.getOutput().shape());
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * kvTime, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * kvTime, 1, 1, headNum * dk);
        this.qLinerLayer.back(queryDelta);
        this.kLinerLayer.back(keyDelta);
        this.vLinerLayer.back(valueDelta);
        Tensor_OP().add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
        Tensor_OP().add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);
        this.diff = qLinerLayer.diff;
    }

    public void diff(Tensor kvDiff) {
        // TODO Auto-generated method stub
        if (dropout) {
            dropoutLayer2.back(delta);
            this.getoLinerLayer().back(dropoutLayer2.diff, oi);
        } else {
            this.getoLinerLayer().back(delta, oi);
        }
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.qLinerLayer.getOutput().shape());
        kt.view(this.kLinerLayer.getOutput().shape());
        vt.view(this.vLinerLayer.getOutput().shape());
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        this.qLinerLayer.back(queryDelta);
        this.kLinerLayer.back(keyDelta);
        this.vLinerLayer.back(valueDelta);
        Tensor_OP().add(kvDiff, this.kLinerLayer.diff, kvDiff);
        Tensor_OP().add(kvDiff, this.vLinerLayer.diff, kvDiff);
        this.diff = qLinerLayer.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 设置输入
         *
         */
        this.setInput(input);
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
    }

    public void forward(Tensor q, Tensor k, Tensor v) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         *
         */
        this.init(q);
        /**
         * 设置输入
         *
         */
        this.setInput(q);
        /**
         * 计算输出
         *
         */
        this.output(q, k, v);
    }

    public void forward(Tensor q, Tensor k, Tensor v, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         *
         */
        this.init(q);
        /**
         * 设置输入
         *
         */
        this.setInput(q);
        this.mask = mask;
        /**
         * 计算输出
         *
         */
        this.output(q, k, v);
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

    public void back(Tensor delta, Tensor kvDiff) {
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
        this.diff(kvDiff);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        qLinerLayer.update();
        kLinerLayer.update();
        vLinerLayer.update();
        oLinerLayer.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mutli_head_attention;
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
        qLinerLayer.saveModel(outputStream);
        kLinerLayer.saveModel(outputStream);
        vLinerLayer.saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        qLinerLayer.loadModel(inputStream);
        kLinerLayer.loadModel(inputStream);
        vLinerLayer.loadModel(inputStream);
        getoLinerLayer().loadModel(inputStream);
    }

    public FullyLayer getoLinerLayer() {
        return oLinerLayer;
    }

    public void setoLinerLayer(FullyLayer oLinerLayer) {
        this.oLinerLayer = oLinerLayer;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}
