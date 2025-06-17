package com.omega.engine.nn.layer.llama;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.gpu.FlashAttentionV2Kernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * CausalSelfAttentionLayer
 *
 * @author Administrator
 */
public class LlamaFlashAttentionLayer extends LlamaAttentionLayer {
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private boolean bias = false;
    private FullyLayer qLinerLayer;
    private FullyLayer kLinerLayer;
    private FullyLayer vLinerLayer;
    //	private FullyLayer qkvLinerLayer;
    private FullyLayer oLinerLayer;
    //	private DropoutLayer dropoutLayer;
    private DropoutLayer dropoutLayer2;
    private AttentionKernel attentionKernel;
    private RoPEKernel ropeKernel;
    private FlashAttentionV2Kernel kernel;
    private Tensor rq;
    private Tensor rk;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor dqt;
    private Tensor dkt;
    private Tensor dvt;
    private Tensor vaccum;
    private Tensor oi;
    private Tensor dvaccum;
    private int batchSize = 1;
    private boolean dropout = false;

    public LlamaFlashAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout) {
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

    public LlamaFlashAttentionLayer(int embedDim, int headNum, int time, boolean bias, boolean dropout, Network network) {
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
        float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
        LlamaFlashAttentionLayer mal = new LlamaFlashAttentionLayer(embedDim, headNum, time, false, false, tf);
        Tensor[] cs = RoPEKernel.getCosAndSin(time, embedDim, headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        //		mal.forward(input);
        for (int i = 0; i < 10; i++) {
            mal.forward(cos, sin, input);
            //			input.showDM();
            //			mal.getWeights().showDM();
            mal.getOutput().showShape();
            mal.getOutput().showDM();
            mal.back(cos, sin, delta);
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
        this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
        //		this.qLinerLayer.weight.showDM();
        this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, this.embedDim * this.embedDim * 0.01f + 0.01f), true);
        //		this.kLinerLayer.weight.showDM();
        this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 2 * this.embedDim * this.embedDim * 0.01f + 0.01f), true);
        //		this.vLinerLayer.weight.showDM();
        this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
        if (kernel == null) {
            kernel = new FlashAttentionV2Kernel(headNum, time, dk, cuda());
        }
        if (this.dropout) {
            //			this.dropoutLayer = new DropoutLayer(0.1f, this.network);
            this.dropoutLayer2 = new DropoutLayer(0.1f, oLinerLayer);
        }
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (ropeKernel == null) {
            ropeKernel = new RoPEKernel(cuda());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.getShape()[0];
        this.time = this.network.time;
        this.batchSize = number / time;
        if (this.vaccum == null || this.vaccum.getShape()[0] != this.batchSize || this.vaccum.getShape()[2] != this.time) {
            // [batch_size，time，head_num，d_k]
            this.rq = Tensor.createGPUTensor(this.rq, batchSize, time, headNum, dk, true);
            this.rk = Tensor.createGPUTensor(this.rk, batchSize, time, headNum, dk, true);
            this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
            this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
            // [batch_size, n_heads, len_q, dim_v]
            this.vaccum = Tensor.createGPUTensor(this.vaccum, batchSize, headNum, time, dk, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
        }
        this.qt.viewOrg();
        this.kt.viewOrg();
        this.vt.viewOrg();
        this.rq.viewOrg();
        this.rk.viewOrg();
        if (this.qLinerLayer.getOutput() != null) {
            this.qLinerLayer.getOutput().viewOrg();
            this.kLinerLayer.getOutput().viewOrg();
            this.vLinerLayer.getOutput().viewOrg();
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dvaccum == null) {
            this.dvaccum = Tensor.createGPUTensor(this.dvaccum, batchSize, headNum, time, dk, true);
            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
        } else {
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
    }

    public void output(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        //		System.out.println("in");
        //		this.input.showDM();
        this.qLinerLayer.forward(this.input);
        this.kLinerLayer.forward(this.input);
        this.vLinerLayer.forward(this.input);
        Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
        /**
         * apply RoPE

         */
        ropeKernel.forward(cos, sin, query, key, rq, rk);
        Tensor_OP().permute(rq, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(rk, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        kernel.forward(qt, kt, vt, vaccum);
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.oLinerLayer.forward(oi);
        this.output = this.oLinerLayer.getOutput();
        if (dropout) {
            dropoutLayer2.forward(this.oLinerLayer.getOutput());
            this.output = dropoutLayer2.getOutput();
        }
        //		this.output.showDMByNumber(0);
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
        if (dropout) {
            dropoutLayer2.back(delta);
            this.oLinerLayer.back(dropoutLayer2.diff, oi);
        } else {
            this.oLinerLayer.back(delta, oi);
        }
        attentionKernel.unpermute_backward(dvaccum, oi, batchSize, time, headNum, dk);
        kernel.backward(qt, kt, vt, vaccum, dvaccum, dqt, dkt, dvt);
        qt.view(this.qLinerLayer.getOutput().shape());
        kt.view(this.kLinerLayer.getOutput().shape());
        vt.view(this.vLinerLayer.getOutput().shape());
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        /**
         * RoPE backward

         */
        ropeKernel.backward(cos, sin, qt, kt, rq, rk);
        Tensor queryDelta = rq.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = rk.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        this.qLinerLayer.back(queryDelta);
        this.kLinerLayer.back(keyDelta);
        this.vLinerLayer.back(valueDelta);
        Tensor_OP().add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
        Tensor_OP().add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);
        this.diff = this.qLinerLayer.diff;
        //		this.diff.showDMByNumber(0);
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
    }

    public void forward(Tensor cos, Tensor sin, Tensor input) {
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
        this.output(cos, sin);
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
        qLinerLayer.update();
        kLinerLayer.update();
        vLinerLayer.update();
        oLinerLayer.update();
    }

    @Override
    public void putParamters() {
        qLinerLayer.putParamters();
        kLinerLayer.putParamters();
        vLinerLayer.putParamters();
        oLinerLayer.putParamters();
    }

    @Override
    public void putParamterGrads() {
        qLinerLayer.putParamterGrads();
        kLinerLayer.putParamterGrads();
        vLinerLayer.putParamterGrads();
        oLinerLayer.putParamterGrads();
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

    @Override
    public void saveModel(RandomAccessFile outputStream) throws IOException {
        // TODO Auto-generated method stub
    }

    @Override
    public void loadModel(RandomAccessFile inputStream) throws IOException {
        // TODO Auto-generated method stub
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

