package com.omega.engine.nn.layer.clip.bert;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

/**
 * BertSelfAttentionLayer
 *
 * @author Administrator
 */
public class BertSelfAttentionLayer extends Layer {
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private FullyLayer qLinerLayer;
    private FullyLayer kLinerLayer;
    private FullyLayer vLinerLayer;
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    private int batchSize = 1;
    private boolean dropout = false;

    public BertSelfAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout) {
        this.bias = bias;
        this.time = time;
        this.embedDim = embedDim;
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

    public BertSelfAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.dropout = dropout;
        this.initLayers();
    }

    public static void makeMask(Tensor mask, int idsLen) {
        for (int i = 0; i < idsLen; i++) {
            mask.data[i] = 0;
        }
        mask.hostToDevice();
    }

    public static void main(String[] args) {
        int embedDim = 24;
        int headNum = 12;
        int batchSize = 1;
        int time = 12;
        Transformer tf = new Transformer();
        tf.RUN_MODEL = RunModel.TEST;
        tf.number = batchSize * time;
        tf.time = time;
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        Tensor mask = new Tensor(batchSize, 1, 1, time, MatrixUtils.val(batchSize * time, -10000.0f), true);
        makeMask(mask, 1 + 2);
        mask.showDM();
        BertSelfAttentionLayer mal = new BertSelfAttentionLayer(embedDim, headNum, time, false, false, tf);
        //		mal.forward(input);
        for (int i = 0; i < 10; i++) {
            input.showDM();
            mal.forward(input, mask);
            mal.getOutput().showDM();
        }
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
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.1f), true);
        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.1f), true);
        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.1f), true);
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(network.cudaManager);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        if (network.CUDNN && softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, network.cudaManager);
        }
        this.number = input.number;
        this.time = this.network.time;
        this.batchSize = this.number / this.time;
        if (network.RUN_MODEL == RunModel.EVAL) {
            if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
                // [batch_size，time，head_num，d_k]
                this.qt = CUDAMemoryManager.getCache("clip-attn-qt", batchSize, headNum, time, dk);
                this.kt = CUDAMemoryManager.getCache("clip-attn-kt", batchSize, headNum, time, dk);
                this.vt = CUDAMemoryManager.getCache("clip-attn-vt", batchSize, headNum, time, dk);
                // [batch_size，n_heads，len_q，len_k]
                if (time < dk) {
                    this.temp = CUDAMemoryManager.getCache("clip-attn-temp1", batchSize, headNum, time, dk);
                } else {
                    this.temp = CUDAMemoryManager.getCache("clip-attn-temp2", batchSize, headNum, time, time);
                }
                // [batch_size，n_heads，len_q，len_k]
                this.attn = CUDAMemoryManager.getCache("clip-attn-attn", batchSize, headNum, time, time);
                // [batch_size, len_q, n_heads * dim_v]
                this.oi = CUDAMemoryManager.getCache("clip-attn-oi", batchSize * time, 1, 1, embedDim);
            }
        } else {
            if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
                // [batch_size，time，head_num，d_k]
                this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
                this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
                this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
                // [batch_size，n_heads，len_q，len_k]
                if (time < dk) {
                    this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
                } else {
                    this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, time, true);
                }
                // [batch_size，n_heads，len_q，len_k]
                this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
                // [batch_size, len_q, n_heads * dim_v]
                this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
            }
        }
        this.qt.viewOrg();
        this.kt.viewOrg();
        this.vt.viewOrg();
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
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
        this.getqLinerLayer().forward(this.input);
        this.getkLinerLayer().forward(this.input);
        this.getvLinerLayer().forward(this.input);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.output = oi;
    }

    public void output(Tensor mask) {
        // TODO Auto-generated method stub
        if (network.RUN_MODEL == RunModel.EVAL) {
            eval(mask);
        } else {
            train(mask);
        }
    }

    public void train(Tensor mask) {
        this.getqLinerLayer().forward(this.input);
        this.getkLinerLayer().forward(this.input);
        this.getvLinerLayer().forward(this.input);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, mask);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.output = oi;
    }

    public void eval(Tensor mask) {
        Tensor qfo = CUDAMemoryManager.getCache("VQVAEAttn_qfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor kfo = CUDAMemoryManager.getCache("VQVAEAttn_kfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor vfo = CUDAMemoryManager.getCache("VQVAEAttn_vfo_cache", batchSize * time, 1, 1, embedDim);
        this.getqLinerLayer().forward(this.input, qfo);
        this.getkLinerLayer().forward(this.input, kfo);
        this.getvLinerLayer().forward(this.input, vfo);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, mask);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.output = oi;
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);
        attentionKernel.addMask(preatt, mask, preatt);
        if (network.CUDNN) {
            softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        } else {
            if (network.RUN_MODEL == RunModel.TEST) {
                attentionKernel.softmax_unmask_test_forward(preatt, attn, batchSize, headNum, time, 1);
            } else {
                attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, headNum, time, 1);
            }
        }
        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        if (network.RUN_MODEL == RunModel.TEST) {
            attentionKernel.softmax_unmask_test_forward(preatt, attn, batchSize, headNum, time, d_k);
            //			attentionKernel.softmax_test_forward(preatt, attn, batchSize, headNum, time, d_k);
        } else {
            attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, headNum, time, d_k);
            //			attentionKernel.softmax_forward(preatt, attn, batchSize, headNum, time, d_k);
        }
        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
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

    public void forward(Tensor input, Tensor mask) {
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
        this.output(mask);
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
        getqLinerLayer().update();
        getkLinerLayer().update();
        getvLinerLayer().update();
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

    //	public Tensor getWeights() {
    //		return weights;
    //	}
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
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getqLinerLayer().loadModel(inputStream);
        getkLinerLayer().loadModel(inputStream);
        getvLinerLayer().loadModel(inputStream);
    }

    public FullyLayer getqLinerLayer() {
        return qLinerLayer;
    }

    public void setqLinerLayer(FullyLayer qLinerLayer) {
        this.qLinerLayer = qLinerLayer;
    }

    public FullyLayer getkLinerLayer() {
        return kLinerLayer;
    }

    public void setkLinerLayer(FullyLayer kLinerLayer) {
        this.kLinerLayer = kLinerLayer;
    }

    public FullyLayer getvLinerLayer() {
        return vLinerLayer;
    }

    public void setvLinerLayer(FullyLayer vLinerLayer) {
        this.vLinerLayer = vLinerLayer;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
    }
}

