package com.omega.engine.nn.layer.t5;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * T5AttentionLayer
 *
 * @author Administrator
 */
public class T5AttentionLayer extends Layer {
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private FullyLayer qLinerLayer;
    private FullyLayer kLinerLayer;
    private FullyLayer vLinerLayer;
    private FullyLayer oLinerLayer;
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    private int batchSize = 1;

    public T5AttentionLayer(int embedDim, int headNum, int time, boolean bias, Network network) {
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
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
        
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

        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

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
        this.number = input.number;
        this.batchSize = this.number / this.time;
        if (network.CUDNN && softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, network.cudaManager);
        }
        if (this.qt != null) {
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
        }
        if (network.RUN_MODEL == RunModel.EVAL) {
            this.qt = CUDAMemoryManager.getCache(network.id+"clip-attn-qt", batchSize, headNum, time, dk);
            this.kt = CUDAMemoryManager.getCache(network.id+"clip-attn-kt", batchSize, headNum, time, dk);
            this.vt = CUDAMemoryManager.getCache(network.id+"clip-attn-vt", batchSize, headNum, time, dk);
            // [batch_size，n_heads，len_q，len_k]
            if (time < dk) {
                this.temp = CUDAMemoryManager.getCache(network.id+"clip-attn-temp1", batchSize, headNum, time, dk);
            } else {
                this.temp = CUDAMemoryManager.getCache(network.id+"clip-attn-temp2", batchSize, headNum, time, time);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = CUDAMemoryManager.getCache(network.id+"clip-attn-attn", batchSize, headNum, time, time);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = CUDAMemoryManager.getCache(network.id+"clip-attn-oi", batchSize * time, 1, 1, embedDim);
            
            this.output = CUDAMemoryManager.getCache(network.id+"clip-attn-output", batchSize * time, 1, 1, embedDim);
            
        } else {
            if (this.qt == null || this.qt.number != this.batchSize) {
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
    }
    
    public void output(Tensor mask) {
        if (network.RUN_MODEL == RunModel.EVAL) {
            eval(mask);
        } else {
            train(mask);
        }
    }
    
    public void eval(Tensor mask) {
        Tensor qfo = CUDAMemoryManager.getCache(network.id+"clip_attn_qfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor kfo = CUDAMemoryManager.getCache(network.id+"clip_attn_kfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor vfo = CUDAMemoryManager.getCache(network.id+"clip_attn_vfo_cache", batchSize * time, 1, 1, embedDim);
        this.getqLinerLayer().forward(this.input, qfo);
        this.getkLinerLayer().forward(this.input, kfo);
        this.getvLinerLayer().forward(this.input, vfo);
//        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
//        query.showDM("q");
//        Tensor_OP().mul(query, d_k, query);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, mask);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi, output);
//        output.showDM("output");
    }

    public void train(Tensor mask) {
        this.getqLinerLayer().forward(this.input);
        this.getkLinerLayer().forward(this.input);
        this.getvLinerLayer().forward(this.input);
//        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
//        Tensor_OP().mul(query, d_k, query);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt, mask);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask) {
        Tensor preatt = temp;
//        key.showDM("key");
        GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
//        preatt.showDM("score");
        if (mask != null) {
            Tensor_OP().add(preatt, mask, preatt);
        }
        if (network.CUDNN) {
            softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        } else {
            float d_k = 1.0f;
            attentionKernel.softmax_unmask_test_forward(preatt, attn, batchSize, headNum, time, d_k);
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
        getoLinerLayer().update();
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

    //	public Tensor getWeights() {
    //		return weights;
    //	}
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
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getqLinerLayer().loadModel(inputStream);
        getkLinerLayer().loadModel(inputStream);
        getvLinerLayer().loadModel(inputStream);
        getoLinerLayer().loadModel(inputStream);
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

