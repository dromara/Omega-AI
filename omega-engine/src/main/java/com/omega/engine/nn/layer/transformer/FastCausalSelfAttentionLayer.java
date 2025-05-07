package com.omega.engine.nn.layer.transformer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.DropoutLayer;
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
 * CausalSelfAttentionLayer
 *
 * @author Administrator
 */
public class FastCausalSelfAttentionLayer extends Layer {
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private FullyLayer qkvLinerLayer;
    private FullyLayer oLinerLayer;
    private DropoutLayer dropoutLayer;
    private DropoutLayer dropoutLayer2;
    private AttentionKernel attentionKernel;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor dqt;
    private Tensor dkt;
    private Tensor dvt;
    private Tensor vaccum;
    private Tensor preatt;
    private Tensor attn;
    private Tensor oi;
    private Tensor dvaccum;
    private Tensor dattn;
    private Tensor dpreatt;
    private Tensor dqkv;
    private int batchSize = 1;
    private boolean dropout = false;

    public FastCausalSelfAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout) {
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

    public FastCausalSelfAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout, Network network) {
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
        this.dropout = dropout;
        this.initLayers();
    }

    public static void main(String[] args) {
        int embedDim = 4;
        int headNum = 2;
        int batchSize = 3;
        int time = 3;
        Transformer tf = new Transformer();
        tf.number = batchSize * time;
        tf.time = time;
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
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        //		input.showDM();
        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        //		float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
        FastCausalSelfAttentionLayer mal = new FastCausalSelfAttentionLayer(embedDim, headNum, time, false, false, tf);
        //		mal.forward(input);
        for (int i = 0; i < 10; i++) {
            mal.forward(input);
            //			input.showDM();
            //			mal.getWeights().showDM();
            mal.getOutput().showShape();
            mal.getOutput().showDM();
            mal.back(delta);
            //			delta.showDM();
            mal.diff.showDM();
            //			delta.copyData(tmp);
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
        this.qkvLinerLayer = new FullyLayer(embedDim, 3 * embedDim, bias, this.network);
        //		NanoGPT net = (NanoGPT) this.network;
        this.qkvLinerLayer.weight = new Tensor(1, 1, embedDim, 3 * embedDim, RandomUtils.uniform(this.embedDim * 3 * this.embedDim, 0.0f, 0.02f), true);
        //		this.qkvLinerLayer.weight = new Tensor(1, 1, embedDim, 3 * embedDim, RandomUtils.order(this.embedDim * 3 * this.embedDim, 0.01f, 0.01f), true);
        //
        //		float[] data = RandomUtils.order(embedDim * 3 * embedDim, 0.01f, 0.01f);
        //		for(int r = 0;r<embedDim;r++) {
        //			for(int c = 0;c<embedDim * 3;c++) {
        //				int batch = c / embedDim;
        //				int col = c % embedDim;
        //				this.qkvLinerLayer.weight.data[r * embedDim * 3 + c] = data[batch * embedDim * embedDim + r * embedDim + col];
        //			}
        //		}
        //
        //		this.qkvLinerLayer.weight.hostToDevice();
        //		PrintUtils.printImage(this.qkvLinerLayer.weight);
        //		this.qkvLinerLayer.weight.showDM();
        this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, (float)(0.02f / Math.sqrt(2 * net.decoderNum))), true);
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
        if (this.dropout) {
            this.dropoutLayer = new DropoutLayer(0.1f, this.network);
            this.dropoutLayer2 = new DropoutLayer(0.1f, oLinerLayer);
        }
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
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
        this.time = this.network.time;
        this.batchSize = number / time;
        if (this.preatt == null || this.preatt.number != this.batchSize || this.preatt.width != this.time) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
            this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            this.preatt = Tensor.createTensor(this.preatt, batchSize, headNum, time, time, true);
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createTensor(this.attn, batchSize, headNum, time, time, true);
            // [batch_size, n_heads, len_q, dim_v]
            this.vaccum = Tensor.createTensor(this.vaccum, batchSize, headNum, time, dk, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createTensor(this.oi, batchSize, time, headNum, dk, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.time = this.network.time;
        this.batchSize = number / time;
        if (this.preatt == null || this.preatt.number != this.batchSize || this.preatt.width != this.time) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
            this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            this.preatt = Tensor.createTensor(this.preatt, batchSize, headNum, time, time, true);
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createTensor(this.attn, batchSize, headNum, time, time, true);
            // [batch_size, n_heads, len_q, dim_v]
            this.vaccum = Tensor.createTensor(this.vaccum, batchSize, headNum, time, dk, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dvaccum == null) {
            this.dvaccum = Tensor.createTensor(this.dvaccum, batchSize, headNum, time, dk, true);
            this.dqt = Tensor.createTensor(this.dqt, batchSize, headNum, time, dk, true);
            this.dkt = Tensor.createTensor(this.dkt, batchSize, headNum, time, dk, true);
            this.dvt = Tensor.createTensor(this.dvt, batchSize, headNum, time, dk, true);
            this.dattn = Tensor.createTensor(this.dattn, batchSize, headNum, time, time, true);
            this.dpreatt = Tensor.createTensor(this.dpreatt, batchSize, headNum, time, time, true);
            this.dqkv = Tensor.createTensor(this.dqkv, number, 1, 1, 3 * embedDim, true);
        } else {
            //			this.dqkv.clearGPU();
            //			this.dvaccum.clearGPU();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		System.out.println("in");
        //		this.input.showDM();
        this.qkvLinerLayer.forward(this.input);
        //		System.err.println("-------------------------");
        //		this.qkvLinerLayer.getOutput().showDM();
        //		System.err.println("-------------------------");
        attentionKernel.permute(this.qkvLinerLayer.getOutput(), qt, kt, vt, batchSize, time, headNum, dk);
        //
        //		qt.showDM();
        //		kt.showDM();
        //		vt.showDM();
        //
        scaledDotProductAttention(qt, kt, vt);
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.oLinerLayer.forward(oi);
        this.output = this.oLinerLayer.getOutput();
        if (dropout) {
            dropoutLayer2.forward(this.oLinerLayer.getOutput());
            this.output = dropoutLayer2.getOutput();
        }
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        //		GPU_OP().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        if (network.RUN_MODEL == RunModel.TEST) {
            attentionKernel.scale(preatt, d_k, batchSize, headNum, time);
            attentionKernel.softmax_test_forward(preatt, attn, batchSize, headNum, time);
        } else {
            attentionKernel.softmax_forward(preatt, attn, batchSize, headNum, time, d_k);
        }
        Tensor tmp = attn;
        if (dropout) {
            dropoutLayer.forward(attn);
            tmp = dropoutLayer.getOutput();
        }
        //		attn.syncHost();
        //		PrintUtils.printImage(attn);
        //		GPU_OP().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;
        if (dropout) {
            tmp = dropoutLayer.getOutput();
        }
        // backward into datt
        //		GPU_OP().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        // backward into dv
        //		GPU_OP().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
        if (dropout) {
            dropoutLayer.back(dattn);
            dattn = dropoutLayer.diff;
        }
        // backward into preatt
        //		float d_k = (float) (1.0f / Math.sqrt(dk));
        //		attentionKernel.softmax_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum, d_k);
        attentionKernel.softmax_test_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum);
        // backward into q
        //		GPU_OP().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        // backward into k
        //		GPU_OP().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
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
            this.oLinerLayer.back(dropoutLayer2.diff, oi);
        } else {
            this.oLinerLayer.back(delta, oi);
        }
        attentionKernel.unpermute_backward(dvaccum, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        attentionKernel.permute_backward(dqkv, dqt, dkt, dvt, batchSize, time, headNum, dk);
        this.qkvLinerLayer.back(dqkv);
        this.diff = this.qkvLinerLayer.diff;
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
        qkvLinerLayer.update();
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
        qkvLinerLayer.saveModel(outputStream);
        oLinerLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        qkvLinerLayer.loadModel(inputStream);
        oLinerLayer.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        qkvLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}

