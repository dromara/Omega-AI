package com.omega.engine.nn.layer.transformer;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.transformer.utils.ENTokenizer;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

/**
 * Multi-Head AttentionLayer
 *
 * @author Administrator
 */
public class MultiHeadAttentionLayer extends Layer {
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private boolean layer_norm = false;
    private FullyLayer qLinerLayer;
    private FullyLayer kLinerLayer;
    private FullyLayer vLinerLayer;
    private FullyLayer oLinerLayer;
    private LNLayer lnLayer;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor scores;
    private Tensor weights;
    private Tensor attn_outputs;
    private Tensor ot;
    private Tensor ro;
    private SoftmaxKernel softmax;
    private int batchSize = 1;

    public MultiHeadAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean layer_norm) {
        this.time = time;
        this.layer_norm = layer_norm;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public MultiHeadAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean layer_norm, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.bias = bias;
        this.layer_norm = layer_norm;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
        int embedDim = 4;
        int headNum = 2;
        int batchSize = 3;
        int time = 5;
        Transformer tf = new Transformer();
        tf.number = batchSize * time;
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
        //		int[] rts = new int[] {2, 3, 3};
        //
        //		for(int b = 0;b<batchSize;b++) {
        //			int rt = rts[b];
        //			for(int t = 0;t<time;t++) {
        //				if(t > rt) {
        //					for(int n = 0;n<embedDim;n++) {
        //						data[b * time * embedDim + t * embedDim + n] = 0;
        //					}
        //				}
        //			}
        //		}
        //		float[] maskData = new float[] {1,1,1,0,0,1,1,1,1,0,1,1,1,1,0};
        //
        //		Tensor mask = new Tensor(batchSize, 1, time, time, maskData, true);
        Tensor mask = ENTokenizer.triu(batchSize, headNum, time, time, 1);
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        //		input.showDM();
        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
        MultiHeadAttentionLayer mal = new MultiHeadAttentionLayer(embedDim, headNum, time, false, true, tf);
        //		mal.forward(input);
        for (int i = 0; i < 10; i++) {
            mal.forward(input, mask);
            //			input.showDM();
            //			mal.getWeights().showDM();
            mal.getOutput().showShape();
            mal.getOutput().showDM();
            mal.back(delta);
            //			delta.showDM();
            mal.diff.showDM();
            delta.copyData(tmp);
        }
    }

    public void initLayers() {
        //		float stdv = (float) (1.0f / Math.sqrt(embedDim));
        //		float stdv = (float) (2.0f / Math.sqrt(embedDim + embedDim));
        this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.0f), true);
        //		Tensor qwt = new Tensor(1, 1, embedDim, embedDim, true);
        //		TensorOP.permute(this.qLinerLayer.weight, qwt, new int[] {0, 1, 3, 2});
        //		this.qLinerLayer.weight = qwt;
        //		this.qLinerLayer.bias = new Tensor(1, 1, 1, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0, stdv), true);
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0, stdv), true);
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.1f), true);
        this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.02f, 0.0f), true);
        //		Tensor kwt = new Tensor(1, 1, embedDim, embedDim, true);
        //		TensorOP.permute(this.kLinerLayer.weight, kwt, new int[] {0, 1, 3, 2});
        //		this.kLinerLayer.weight = kwt;
        //		this.kLinerLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.uniform(this.hiddenSize * this.hiddenSize, 0, stdv), true);
        //		this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.2f), true);
        this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.03f, 0.0f), true);
        //		Tensor vwt = new Tensor(1, 1, embedDim, embedDim, true);
        //		TensorOP.permute(this.vLinerLayer.weight, vwt, new int[] {0, 1, 3, 2});
        //		this.vLinerLayer.weight = vwt;
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.01f), true);
        this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.04f, 0.0f), true);
        //		Tensor owt = new Tensor(1, 1, embedDim, embedDim, true);
        //		TensorOP.permute(this.oLinerLayer.weight, owt, new int[] {0, 1, 3, 2});
        //		this.oLinerLayer.weight = owt;
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.02f), true);
        if (this.layer_norm) {
            this.lnLayer = new LNLayer(this.oLinerLayer);
        }
        if (softmax == null) {
            softmax = new SoftmaxKernel(cuda());
        }
        //		System.out.println(JsonUtils.toJson(this.inputLayer.weight.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.inputLayer.bias.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.selfLayer.weight.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.selfLayer.bias.syncHost()));
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        this.dk = embedDim / headNum;
        this.batchSize = number / time;
        if (this.qt == null || this.qt.number != this.batchSize) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
            this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            this.scores = Tensor.createTensor(this.scores, batchSize, headNum, time, time, true);
            // [batch_size，n_heads，len_q，len_k]
            this.weights = Tensor.createTensor(this.weights, batchSize, headNum, time, time, true);
            // [batch_size, n_heads, len_q, dim_v]
            this.attn_outputs = Tensor.createTensor(this.attn_outputs, batchSize, headNum, time, dk, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.ot = Tensor.createTensor(this.ot, batchSize, time, headNum, dk, true);
            this.ro = Tensor.createTensor(this.ro, batchSize * time, 1, 1, embedDim, true);
        }
        resize();
    }

    public void resize() {
        this.qt.viewOrg();
        this.kt.viewOrg();
        this.vt.viewOrg();
        this.scores.viewOrg();
        this.weights.viewOrg();
        this.attn_outputs.viewOrg();
        this.ot.viewOrg();
        if (this.qLinerLayer.getOutput() != null) {
            this.qLinerLayer.getOutput().viewOrg();
            this.kLinerLayer.getOutput().viewOrg();
            this.vLinerLayer.getOutput().viewOrg();
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        //		if(this.cache_delta == null || output.number != cache_delta.number){
        //			this.cache_delta = new Tensor(number, output.channel, output.height, output.width, true);
        //		}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		System.out.println("in");
        this.qLinerLayer.forward(this.input);
        this.kLinerLayer.forward(this.input);
        this.vLinerLayer.forward(this.input);
        Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, null);
        Tensor_OP().permute(attn_outputs, ot, new int[]{0, 2, 1, 3});
        ot.view(batchSize * time, 1, 1, headNum * dk);
        this.oLinerLayer.forward(ot);
        //		this.oLinerLayer.getOutput().showDM();
        Tensor_OP().add(this.oLinerLayer.getOutput(), this.input, this.ro);
        if (this.layer_norm) {
            this.lnLayer.forward(this.ro);
            this.output = this.lnLayer.getOutput();
        } else {
            this.output = this.ro;
        }
        //		System.out.println("in2");
    }

    public void output(Tensor mask) {
        // TODO Auto-generated method stub
        this.qLinerLayer.forward(this.input);
        this.kLinerLayer.forward(this.input);
        this.vLinerLayer.forward(this.input);
        Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, mask);
        Tensor_OP().permute(attn_outputs, ot, new int[]{0, 2, 1, 3});
        ot.view(batchSize * time, 1, 1, headNum * dk);
        this.oLinerLayer.forward(ot);
        Tensor_OP().add(this.oLinerLayer.getOutput(), this.input, this.ro);
        if (this.layer_norm) {
            this.lnLayer.forward(ro);
            this.output = this.lnLayer.getOutput();
        } else {
            this.output = ro;
        }
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        GPU_OP().bmm(query.getGpuData(), key.getGpuData(), scores.getGpuData(), query.number * query.channel, query.height, key.height, query.width, CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
        //		scores.showShape();
        //		System.out.println("scores.showDM():");
        //		scores.showDM();
        if (mask != null) {
            softmax.softmaxMask(scores, mask, weights, -1e9f);
        } else {
            softmax.softmax(scores, weights);
        }
        //		System.out.println("weights.showDM():");
        //		weights.showDM();
        GPU_OP().bmm(weights.getGpuData(), value.getGpuData(), attn_outputs.getGpuData(), weights.number * weights.channel, weights.height, value.width, weights.width, CUBLAS_OP_N, CUBLAS_OP_N, 1.0f, 0.0f);
    }

    public void scaledDotProductAttentionBackward(Tensor query, Tensor key, Tensor value, Tensor delta, Tensor diffQ, Tensor diffK, Tensor diffV) {
        // vt_diff = weightsT * delta
        diffV.view(value.shape());
        GPU_OP().bmm(weights.getGpuData(), delta.getGpuData(), diffV.getGpuData(), weights.number * weights.channel, weights.width, delta.width, weights.height, CUBLAS_OP_T, CUBLAS_OP_N, 1.0f, 0.0f);
        // weights_diff = delta * vt
        GPU_OP().bmm(delta.getGpuData(), value.getGpuData(), scores.getGpuData(), delta.number * delta.channel, delta.height, value.height, value.width, CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
        // scores_diff = softmax_backward
        softmax.backward_noloss(weights, scores, scores);
        //		scores.showDM();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        //		TensorOP.mul(scores, d_k, scores);
        // kt_diff = deltaT / sqrt(dk) * qt
        diffK.view(key.shape());
        GPU_OP().bmm(scores.getGpuData(), query.getGpuData(), diffK.getGpuData(), scores.number * scores.channel, scores.width, query.width, scores.height, CUBLAS_OP_T, CUBLAS_OP_N, d_k, 0.0f);
        // qt_diff = delta / sqrt(dk) * kt
        diffQ.view(query.shape());
        GPU_OP().bmm(scores.getGpuData(), key.getGpuData(), diffQ.getGpuData(), scores.number * scores.channel, scores.height, key.width, scores.width, CUBLAS_OP_N, CUBLAS_OP_N, d_k, 0.0f);
        //
        //		System.out.println("");
        //		diffV.showDM();
        //		diffK.showDM();
        //		diffQ.showDM();
    }

    public void scaledDotProductAttentionBackward(Tensor query, Tensor key, Tensor value, Tensor delta) {
        // vt_diff = weightsT * delta
        GPU_OP().bmm(weights.getGpuData(), delta.getGpuData(), value.getGrad().getGpuData(), weights.number * weights.channel, weights.width, delta.width, weights.height, CUBLAS_OP_T, CUBLAS_OP_N, 1.0f, 0.0f);
        // weights_diff = delta * vt
        GPU_OP().bmm(delta.getGpuData(), value.getGpuData(), scores.getGpuData(), delta.number * delta.channel, delta.height, weights.width, delta.width, CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
        // scores_diff = softmax_backward
        softmax.backward_noloss(weights, scores, scores);
        float d_k = (float) (1.0f / Math.sqrt(dk));
        //		TensorOP.mul(scores, d_k, scores);
        // kt_diff = deltaT / sqrt(dk) * qt
        GPU_OP().bmm(scores.getGpuData(), query.getGpuData(), key.getGrad().getGpuData(), scores.number * scores.channel, scores.width, query.width, scores.height, CUBLAS_OP_T, CUBLAS_OP_N, d_k, 0.0f);
        // qt_diff = delta / sqrt(dk) * kt
        GPU_OP().bmm(scores.getGpuData(), key.getGpuData(), query.getGrad().getGpuData(), scores.number * scores.channel, scores.height, key.width, scores.width, CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        if (this.layer_norm) {
            //			baseKernel.copy_gpu(delta, this.cache_delta, delta.getDataLength(), 1, 1);
            this.lnLayer.back(this.delta);
            this.oLinerLayer.back(this.lnLayer.diff, ot);
        } else {
            this.oLinerLayer.back(delta, ot);
        }
        ot.view(batchSize, time, headNum, dk);
        Tensor_OP().permute(ot, attn_outputs, new int[]{0, 2, 1, 3});
        int[] qo_shape = this.qLinerLayer.getOutput().shape();
        int[] ko_shape = this.kLinerLayer.getOutput().shape();
        int[] vo_shape = this.vLinerLayer.getOutput().shape();
        scaledDotProductAttentionBackward(qt, kt, vt, attn_outputs, this.qLinerLayer.getOutput(), this.kLinerLayer.getOutput(), this.vLinerLayer.getOutput());
        qt.view(qo_shape);
        kt.view(ko_shape);
        vt.view(vo_shape);
        Tensor_OP().permute(this.qLinerLayer.getOutput(), qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(this.kLinerLayer.getOutput(), kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(this.vLinerLayer.getOutput(), vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        this.qLinerLayer.back(queryDelta);
        this.kLinerLayer.back(keyDelta);
        this.vLinerLayer.back(valueDelta);
        Tensor_OP().add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
        Tensor_OP().add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);
        Tensor_OP().add(this.qLinerLayer.diff, this.lnLayer.diff, this.qLinerLayer.diff);
        this.diff = this.qLinerLayer.diff;
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

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(mask);
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
        qLinerLayer.update();
        kLinerLayer.update();
        vLinerLayer.update();
        oLinerLayer.update();
        if (layer_norm) {
            lnLayer.update();
        }
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

    public Tensor getWeights() {
        return weights;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
        if (layer_norm) {
            lnLayer.accGrad(scale);
        }
    }
}

