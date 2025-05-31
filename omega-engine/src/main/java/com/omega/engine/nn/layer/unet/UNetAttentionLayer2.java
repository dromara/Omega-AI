package com.omega.engine.nn.layer.unet;

import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
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
public class UNetAttentionLayer2 extends Layer {
    public LNLayer norm;
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    private int time;
    private int kvTime;
    private int headNum = 1;
    private int embedDim = 0;
    private int kDim = 0;
    private int vDim = 0;
    private int dk = 0;
    //	public GNLayer gn;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;
    private boolean residualConnect = false;
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
    private int batchSize = 1;
    private boolean dropout = false;

    public UNetAttentionLayer2(int embedDim, int kDim, int vDim, int headNum, int time, int kvTime, boolean bias, boolean dropout, boolean residualConnect) {
        this.bias = bias;
        this.residualConnect = residualConnect;
        this.time = time;
        this.kvTime = kvTime;
        this.embedDim = embedDim;
        this.kDim = kDim;
        this.vDim = vDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = time;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
        this.initLayers();
    }

    public UNetAttentionLayer2(int embedDim, int kDim, int vDim, int headNum, int time, int kvTime, boolean bias, boolean dropout, boolean residualConnect, Network network) {
        this.bias = bias;
        this.residualConnect = residualConnect;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.kvTime = kvTime;
        this.embedDim = embedDim;
        this.kDim = kDim;
        this.vDim = vDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = time;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
        this.initLayers();
    }

    public UNetAttentionLayer2(int embedDim, int headNum, int time, boolean bias, boolean dropout, boolean residualConnect, Network network) {
        this.bias = bias;
        this.residualConnect = residualConnect;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.kvTime = time;
        this.embedDim = embedDim;
        this.kDim = embedDim;
        this.vDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.[embedDim:" + embedDim + ",headNum:" + headNum + "]");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = time;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
        this.initLayers();
    }

    public static void main(String[] args) {
        //		Transformer tf = new Transformer();
        //
        //		int embedDim = 4;
        //		int kDim = 6;
        //		int vDim = 6;
        //		int headNum = 2;
        //		int kvTime = 12;
        //		int channel = 4;
        //		int height = 8;
        //		int width = 8;
        //
        //		int N = 2;
        //
        //		UNetAttentionLayer2 layer = new UNetAttentionLayer2(embedDim, kDim, vDim, headNum, kvTime, channel, height, width, 0, false, false, true, tf);
        //
        //		Tensor x = new Tensor(N, channel, height, width, MatrixUtils.order(N * channel * height * width, 0.01f, 0.1f), true);
        //
        //		Tensor context = new Tensor(N * kvTime, 1, 1, vDim, MatrixUtils.order(N * kvTime * vDim, 0.01f, 0.1f), true);
        //
        //		x.showDM();
        //		context.showDM();
        //
        //		layer.forward(x, context, context);
        //		layer.getOutput().showDM();
        //
        //		Tensor delta = new Tensor(N, channel, height, width, MatrixUtils.order(N * channel * height * width, 0.01f, 0.1f), true);
        //
        //		Tensor kvDiff = new Tensor(N * kvTime, 1, 1, vDim, true);
        //
        //		layer.back(delta, kvDiff);
        //
        //		layer.diff.showDM();
        //
        //		kvDiff.showDM();
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
        norm = new LNLayer(this, BNType.fully_bn, 1, 1, width);
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, false, this.network));
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(embedDim * embedDim, 0.1f, 0.01f), true);
        this.setkLinerLayer(new FullyLayer(kDim, embedDim, false, this.network));
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, kDim, MatrixUtils.order(embedDim * kDim, 0.1f, 0.01f), true);
        this.setvLinerLayer(new FullyLayer(vDim, embedDim, false, this.network));
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, vDim, MatrixUtils.order(embedDim * vDim, 0.1f, 0.01f), true);
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(embedDim * embedDim, 0.1f, 0.01f), true);
        if (this.dropout) {
            this.dropoutLayer = new DropoutLayer(0.1f, this.network);
            this.dropoutLayer2 = new DropoutLayer(0.1f, getoLinerLayer());
        }
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(kvTime, 1, 1, cuda());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number;
        if (this.qt != null) {
            //			JCuda.cudaDeviceSynchronize();
            this.output.viewOrg();
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
            this.oi.viewOrg();
            temp.clearGPU();
        }
        if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
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
            this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
            //			this.output.showShape("output");
        }
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
            this.getoLinerLayer().getOutput().viewOrg();
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
        } else {
            dattn.viewOrg();
            dqt.viewOrg();
            dkt.viewOrg();
            dvt.viewOrg();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        Tensor x = this.input; //[b, h*w, c]
        x = x.view(batchSize * time, 1, 1, width);
        norm.forward(x);
        x = norm.getOutput();
        this.getqLinerLayer().forward(x);
        this.getkLinerLayer().forward(x);
        this.getvLinerLayer().forward(x);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, kvTime, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, kvTime, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        //		qt.showDM("qt");
        //		kt.showDM("kt");
        //		vt.showDM("vt");
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi);
        Tensor out = this.getoLinerLayer().getOutput();
        if (dropout) {
            dropoutLayer2.forward(this.getoLinerLayer().getOutput());
            out = dropoutLayer2.getOutput();
        }
        out.view(batchSize, time, 1, width);
        this.input.view(batchSize, time, 1, width);
        if (residualConnect) {
            Tensor_OP().add(this.input, out, this.output);
        }
    }

    public void output(Tensor k, Tensor v) {
        // TODO Auto-generated method stub
        Tensor x = this.input;
        x = x.view(batchSize * time, 1, 1, channel);
        norm.forward(x);
        x = norm.getOutput();
        //		x.showDM("cross-attn-norm");
        //		System.err.println("kDim:"+kDim);
        //		k.showShape("context.shape");
        this.getqLinerLayer().forward(x);
        this.getkLinerLayer().forward(k);
        this.getvLinerLayer().forward(v);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, kvTime, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, kvTime, headNum, dk);
        //		key.showDM("key");
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        //		this.getqLinerLayer().weight.showDM();
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        //		vaccum.showDM("vaccum");
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        //		oi.showDMByOffset(0, 100, "oi");
        this.getoLinerLayer().forward(oi);
        Tensor out = this.getoLinerLayer().getOutput();
        if (dropout) {
            dropoutLayer2.forward(this.getoLinerLayer().getOutput());
            out = dropoutLayer2.getOutput();
        }
        out.view(batchSize, time, 1, width);
        this.input.view(batchSize, time, 1, width);
        if (residualConnect) {
            Tensor_OP().add(this.input, out, this.output);
        }
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, key.getGpuData(), dk, kvTime * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);
        softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        Tensor tmp = attn;
        if (dropout) {
            dropoutLayer.forward(attn);
            tmp = dropoutLayer.getOutput();
        }
        //		value.showDM();
        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, kvTime, 1.0f, value.getGpuData(), dk, kvTime * dk, tmp.getGpuData(), kvTime, time * kvTime, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;
        if (dropout) {
            tmp = dropoutLayer.getOutput();
        }
        Tensor dvaccum = temp;
        /**
         * backward into dattn[b, nh, t, t2]
         * vt[b, nh, t2, dk] -> [b, nh, dk, t2]
         * dvaccum[b, nh, t, dk]

         */
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, kvTime, time, dk, 1.0f, vt.getGpuData(), dk, kvTime * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), kvTime, time * kvTime, batchSize * headNum);
        /**
         * backward into dvt[b, nh, t2, dk]
         * dvaccum[b, nh, t, dk]
         * attn[b, nh, t, t2] -> [b, nh, t2, t]

         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, kvTime, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), kvTime, time * kvTime, 0.0f, dvt.getGpuData(), dk, kvTime * dk, batchSize * headNum);
        if (dropout) {
            dropoutLayer.back(dattn);
            dattn = dropoutLayer.diff;
        }
        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        //		dattn.showDM();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        /**
         * backward into dqt

         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, kvTime, 1.0f, kt.getGpuData(), dk, kvTime * dk, dpreatt.getGpuData(), kvTime, time * kvTime, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        /**
         * backward into dkt

         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, kvTime, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), kvTime, time * kvTime, 0.0f, dkt.getGpuData(), dk, kvTime * dk, batchSize * headNum);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        delta.view(batchSize * time, 1, 1, channel);
        if (dropout) {
            dropoutLayer2.back(delta);
            this.getoLinerLayer().back(dropoutLayer2.diff, oi);
        } else {
            this.getoLinerLayer().back(delta, oi);
        }
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.getqLinerLayer().getOutput().shape());
        kt.view(this.getkLinerLayer().getOutput().shape());
        vt.view(this.getvLinerLayer().getOutput().shape());
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        this.getqLinerLayer().back(queryDelta);
        this.getkLinerLayer().back(keyDelta);
        this.getvLinerLayer().back(valueDelta);
        Tensor_OP().add(this.getqLinerLayer().diff, this.getkLinerLayer().diff, this.getqLinerLayer().diff);
        Tensor_OP().add(this.getqLinerLayer().diff, this.getvLinerLayer().diff, this.getqLinerLayer().diff);
        // dxt
        Tensor dxt = this.getqLinerLayer().diff;
        norm.input.view(batchSize * time, 1, 1, channel);
        norm.back(dxt);
        if (residualConnect) {
            Tensor_OP().add(this.delta, norm.diff, this.delta);
        }
        delta.viewOrg();
        norm.input.viewOrg();
        this.diff = delta;
    }

    public void diff(Tensor kvDiff) {
        // TODO Auto-generated method stub
        delta.view(batchSize * time, 1, 1, channel);
        if (dropout) {
            dropoutLayer2.back(delta);
            this.getoLinerLayer().back(dropoutLayer2.diff, oi);
        } else {
            this.getoLinerLayer().back(delta, oi);
        }
        //		Tensor oi = this.getoLinerLayer().diff;
        //		oi.showDM("oDiff");
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.getqLinerLayer().getOutput().shape());
        kt.view(this.getkLinerLayer().getOutput().shape());
        vt.view(this.getvLinerLayer().getOutput().shape());
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * kvTime, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * kvTime, 1, 1, headNum * dk);
        this.getqLinerLayer().back(queryDelta);
        this.getkLinerLayer().back(keyDelta);
        this.getvLinerLayer().back(valueDelta);
        Tensor_OP().add(this.getkLinerLayer().diff, this.getvLinerLayer().diff, kvDiff);
        // dxt
        Tensor dxt = this.getqLinerLayer().diff;
        norm.input.view(batchSize * time, 1, 1, channel);
        norm.back(dxt);
        if (residualConnect) {
            Tensor_OP().add(this.delta, norm.diff, this.delta);
        }
        delta.viewOrg();
        norm.input.viewOrg();
        this.diff = delta;
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

    public void forward(Tensor input, Tensor key, Tensor value) {
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
        this.output(key, value);
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

    public void back(Tensor delta, Tensor kvDiff) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(kvDiff);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        norm.update();
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
    	norm.saveModel(outputStream);
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream);
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
        norm.accGrad(scale);
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}

