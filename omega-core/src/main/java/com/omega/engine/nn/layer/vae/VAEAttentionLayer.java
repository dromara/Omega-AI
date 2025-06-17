package com.omega.engine.nn.layer.vae;

import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
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
public class VAEAttentionLayer extends Layer {
    private int groups = 0;
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;
    private boolean residualConnect = false;
    private GNLayer gn;
    private FullyLayer qLinerLayer;
    private FullyLayer kLinerLayer;
    private FullyLayer vLinerLayer;
    private FullyLayer oLinerLayer;
    private DropoutLayer dropoutLayer;
    private DropoutLayer dropoutLayer2;
    private AttentionKernel attentionKernel;
    private Tensor xt;
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

    public VAEAttentionLayer(int embedDim, int headNum, int time, int channel, int height, int width, boolean bias, boolean dropout, boolean residualConnect) {
        this.bias = bias;
        this.residualConnect = residualConnect;
        this.time = time;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
        this.initLayers();
    }

    public VAEAttentionLayer(int embedDim, int headNum, int time, int channel, int height, int width, boolean bias, boolean dropout, boolean residualConnect, Network network) {
        this.bias = bias;
        this.residualConnect = residualConnect;
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
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
        this.initLayers();
    }

    public VAEAttentionLayer(int embedDim, int headNum, int time, int channel, int height, int width, int groups, boolean bias, boolean dropout, boolean residualConnect, Network network) {
        this.bias = bias;
        this.groups = groups;
        this.residualConnect = residualConnect;
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
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.dropout = dropout;
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
        if (groups > 0) {
            gn = new GNLayer(groups, this, BNType.conv_bn);
        }
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        if (this.dropout) {
            this.dropoutLayer = new DropoutLayer(0.1f, this.network);
            this.dropoutLayer2 = new DropoutLayer(0.1f, getoLinerLayer());
        }
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.getShape()[0];
        this.batchSize = this.number;
        if (this.qt == null || this.qt.getShape()[0] != this.batchSize || this.qt.getShape()[2] != this.time) {
            // [batch_size，time，head_num，d_k]
            this.xt = Tensor.createGPUTensor(this.xt, batchSize, time, 1, channel, true);
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
            this.output = Tensor.createGPUTensor(this.output, batchSize, channel, height, width, true);
        }
        this.output.viewOrg();
        this.qt.viewOrg();
        this.kt.viewOrg();
        this.vt.viewOrg();
        this.xt.viewOrg();
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        Tensor x = this.input;
        if (gn != null) {
            gn.forward(x);
            x = gn.getOutput();
        }
        x = x.view(batchSize, channel, 1, height * width);
        // B,C,HW ==> B,HW,C
        Tensor_OP().permute(x, xt, new int[]{0, 3, 2, 1});
        xt = xt.view(batchSize * time, 1, 1, channel);
        this.getqLinerLayer().forward(xt);
        this.getkLinerLayer().forward(xt);
        this.getvLinerLayer().forward(xt);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi);
        Tensor out = this.getoLinerLayer().getOutput();
        if (dropout) {
            dropoutLayer2.forward(this.getoLinerLayer().getOutput());
            out = dropoutLayer2.getOutput();
        }
        this.output.view(batchSize, channel, 1, time);
        if (residualConnect) {
            Tensor_OP().permuteAdd(out, this.output, new int[]{0, 3, 2, 1}); //B,HW,C ==> B,C,HW
        } else {
            Tensor_OP().permute(out, this.output, new int[]{0, 3, 2, 1}); //B,HW,C ==> B,C,HW
        }
        this.output.viewOrg();
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
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
        //		value.showDM();
        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;
        if (dropout) {
            tmp = dropoutLayer.getOutput();
        }
        Tensor dvaccum = temp;
        // backward into datt
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        // backward into dv
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
        if (dropout) {
            dropoutLayer.back(dattn);
            dattn = dropoutLayer.diff;
        }
        // backward into preatt
        float d_k = (float) (1.0f / Math.sqrt(dk));
        //		attentionKernel.softmax_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum, d_k);
        attentionKernel.softmax2_backward(dattn, attn, batchSize, time, embedDim, headNum, d_k);
        Tensor dpreatt = dattn;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        // backward into k
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
        // B,C,H,W ==> B,HW,C
        this.output.view(batchSize, height, width, channel);
        Tensor_OP().permute(delta, this.output, new int[]{0, 2, 3, 1});
        this.output.view(batchSize, time, 1, channel);
        if (dropout) {
            dropoutLayer2.back(this.output);
            this.getoLinerLayer().back(dropoutLayer2.diff, oi);
        } else {
            this.getoLinerLayer().back(this.output, oi);
        }
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.getqLinerLayer().getOutput().shape());
        kt.view(this.getqLinerLayer().getOutput().shape());
        vt.view(this.getqLinerLayer().getOutput().shape());
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
        // B,HW,C ==> B,C,H,W
        xt = xt.view(batchSize, channel, 1, time);
        Tensor_OP().permute(dxt, xt, new int[]{0, 3, 2, 1});
        xt = xt.view(batchSize, channel, height, width);
        if (gn != null) {
            gn.back(xt);
            this.diff = gn.diff;
        } else {
            this.diff = xt;
        }
        if (residualConnect) {
            Tensor_OP().add(this.diff, this.delta, this.diff);
        }
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
        if (gn != null) {
            gn.update();
        }
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

