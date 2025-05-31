package com.omega.engine.nn.layer.vqvae.tiny;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
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
public class VQVAEAttentionLayer extends Layer {
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
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
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

    public VQVAEAttentionLayer(int embedDim, int headNum, int height, int width, int groups, boolean bias, boolean residualConnect, Network network) {
        this.bias = bias;
        this.groups = groups;
        this.residualConnect = residualConnect;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = height * width;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = embedDim;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.initLayers();
    }

    public static void main(String[] args) {
        try {
            int N = 2;
            int headNum = 4;
            int groups = 2;
            int channel = 4;
            int height = 8;
            int width = 8;
            int dataSize = N * channel * height * width;
            Transformer network = new Transformer();
            network.CUDNN = true;
            VQVAEAttentionLayer attn = new VQVAEAttentionLayer(channel, headNum, height, width, groups, false, true, network);
            Tensor x = new Tensor(N, channel, height, width, MatrixUtils.order(dataSize, 0.1f, 0.1f), true);
            Tensor delta = new Tensor(N, channel, height, width, MatrixUtils.order(dataSize, 0.1f, 0.1f), true);
            attn.forward(x);
            attn.getOutput().showDM();
            attn.back(delta);
            attn.diff.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
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
        if (groups > 0) {
            gn = new GNLayer(groups, channel, height, width, BNType.conv_bn, this);
        }
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(qLinerLayer.weight.dataLength, 0.1f, 0.1f), true);
        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(kLinerLayer.weight.dataLength, 0.1f, 0.1f), true);
        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(vLinerLayer.weight.dataLength, 0.1f, 0.1f), true);
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        //		oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(oLinerLayer.weight.dataLength, 0.1f, 0.1f), true);
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
        this.number = input.number;
        this.batchSize = this.number;
        if (network.CUDNN && softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
        }
        if (this.qt != null) {
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
            this.xt.viewOrg();
            this.oi.viewOrg();
        }
        if (network.RUN_MODEL == RunModel.EVAL) {
            // [batch_size，time，head_num，d_k]
            this.xt = CUDAMemoryManager.getCache("attn-xt", batchSize, time, 1, channel);
            this.qt = CUDAMemoryManager.getCache("attn-qt", batchSize, headNum, time, dk);
            this.kt = CUDAMemoryManager.getCache("attn-kt", batchSize, headNum, time, dk);
            this.vt = CUDAMemoryManager.getCache("attn-vt", batchSize, headNum, time, dk);
            // [batch_size，n_heads，len_q，len_k]
            if (time < dk) {
                this.temp = CUDAMemoryManager.getCache("attn-temp1", batchSize, headNum, time, dk);
            } else {
                this.temp = CUDAMemoryManager.getCache("attn-temp2", batchSize, headNum, time, time);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = CUDAMemoryManager.getCache("attn-attn", batchSize, headNum, time, time);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = CUDAMemoryManager.getCache("attn-oi", batchSize * time, 1, 1, embedDim);
        } else {
            if (this.qt == null || this.qt.number != this.batchSize) {
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
        }
        if (this.output != null) {
            this.output.viewOrg();
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
            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
        }
        this.dqt = this.getqLinerLayer().getOutput().view(batchSize, headNum, time, dk);
        this.dkt = this.getkLinerLayer().getOutput().view(batchSize, headNum, time, dk);
        this.dvt = this.getvLinerLayer().getOutput().view(batchSize, headNum, time, dk);
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        if (network.RUN_MODEL == RunModel.EVAL) {
            output_eval();
        } else {
            train();
        }
    }

    public void train() {
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
        out.view(batchSize, time, 1, channel);
        this.output.view(batchSize, channel, 1, time);
        Tensor_OP().permute(out, this.output, new int[]{0, 3, 2, 1}); //B,HW,C ==> B,C,HW
        if (residualConnect) {
            Tensor_OP().add(this.input, this.output, this.output);
        }
        this.output.viewOrg();
    }

    public void output_eval() {
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
        Tensor qfo = CUDAMemoryManager.getCache("VQVAEAttn_qfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor kfo = CUDAMemoryManager.getCache("VQVAEAttn_kfo_cache", batchSize * time, 1, 1, embedDim);
        Tensor vfo = CUDAMemoryManager.getCache("VQVAEAttn_vfo_cache", batchSize * time, 1, 1, embedDim);
        this.getqLinerLayer().forward(xt, qfo);
        this.getkLinerLayer().forward(xt, kfo);
        this.getvLinerLayer().forward(xt, vfo);
        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
        this.getoLinerLayer().forward(oi, xt);
        Tensor out = this.getoLinerLayer().getOutput();
        out.view(batchSize, time, 1, channel);
        this.output = x;
        this.output.view(batchSize, channel, 1, time);
        Tensor_OP().permute(out, this.output, new int[]{0, 3, 2, 1}); //B,HW,C ==> B,C,HW
        if (residualConnect) {
            Tensor_OP().add(this.input, this.output, this.output);
        }
        this.output.viewOrg();
    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        if (network.CUDNN) {
            attentionKernel.scale(preatt, d_k, batchSize, headNum, time);
            softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        } else {
            if (network.RUN_MODEL == RunModel.TEST) {
                //				attentionKernel.scale(preatt, d_k, batchSize, headNum, time);
                attentionKernel.softmax_unmask_test_forward(preatt, attn, batchSize, headNum, time, d_k);
            } else {
                attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, headNum, time, d_k);
            }
        }
        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;
        Tensor dvaccum = temp;
        // backward into datt
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        // backward into dv
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
        Tensor dpreatt = temp;
        float d_k = (float) (1.0f / Math.sqrt(dk));
        // backward into preatt
        if (network.CUDNN) {
            softmaxKernel.softmax_backward(attn, dattn, dattn);
            Tensor_OP().mul(dattn, d_k, dattn);
            dpreatt = dattn;
        } else {
            attentionKernel.softmax_unmask_backward(dpreatt, dattn, attn, batchSize, time, headNum, d_k);
        }
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
        this.getoLinerLayer().back(this.output, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(batchSize, time, headNum, dk);
        kt.view(batchSize, time, headNum, dk);
        vt.view(batchSize, time, headNum, dk);
        Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        //		qt.showDMByOffset(0, 10, "qt");
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor qDiff = this.getqLinerLayer().getOutput().viewOrg();
        Tensor kDiff = this.getkLinerLayer().getOutput().viewOrg();
        Tensor vDiff = this.getvLinerLayer().getOutput().viewOrg();
        this.getqLinerLayer().back(queryDelta, qDiff);
        this.getkLinerLayer().back(keyDelta, kDiff);
        this.getvLinerLayer().back(valueDelta, vDiff);
        Tensor_OP().add(qDiff, kDiff, qDiff);
        Tensor_OP().add(qDiff, vDiff, qDiff);
        // dxt
        Tensor dxt = qDiff;
        dxt.view(batchSize, time, 1, channel);
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
        dxt.viewOrg();
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
        //		input.showDMByOffset(0, 100, "123");
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
        if (groups > 0) {
            gn.saveModel(outputStream);
        }
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        if (groups > 0) {
            gn.loadModel(inputStream);
        }
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
        if (groups > 0) {
            gn.accGrad(scale);
        }
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}

