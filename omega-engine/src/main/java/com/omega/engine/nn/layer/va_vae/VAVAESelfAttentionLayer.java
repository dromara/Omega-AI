package com.omega.engine.nn.layer.va_vae;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * SelfAttentionLayer
 *
 * @author Administrator
 */
public class VAVAESelfAttentionLayer extends Layer {
	
    public ConvolutionLayer qLinerLayer;
    public ConvolutionLayer kLinerLayer;
    public ConvolutionLayer vLinerLayer;
    public ConvolutionLayer oLinerLayer;
    private int channel;
    private int height;
    private int width;
    private int time;
    private int embedDim = 0;
    private boolean bias = false;
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private Tensor qt;
//    private Tensor kt;
//    private Tensor vt;
    private Tensor dqkvt;
    private Tensor temp;
    private Tensor attn;
//    private Tensor oi;
    private Tensor dattn;
    private int batchSize = 1;

    public VAVAESelfAttentionLayer(int channel, int height, int width, boolean bias) {
        this.bias = bias;
        this.time = height * width;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public VAVAESelfAttentionLayer(int channel, int height, int width, boolean bias, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = height * width;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
//        int embedDim = 64;
//        int headNum = 8;
//        int batchSize = 2;
//        int time = 512;
//        Transformer tf = new Transformer();
//        tf.number = batchSize * time;
//        tf.time = time;
//        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
//        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
//        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
//        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
//        SDVAESelfConvAttentionLayer mal = new SDVAESelfConvAttentionLayer(embedDim, headNum, time, false, tf);
//        for (int i = 0; i < 10; i++) {
//            mal.forward(input);
//            mal.getOutput().showShape();
//            mal.getOutput().showDM();
//            mal.back(delta);
//            mal.diff.showDM();
//        }
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
    	this.qLinerLayer = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, bias, network);
        this.kLinerLayer = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, bias, network);
        this.vLinerLayer = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, bias, network);

        this.oLinerLayer = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, bias, network);

        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = input.number;
        if (this.qt != null) {
            this.qt.viewOrg();
//            this.kt.viewOrg();
//            this.vt.viewOrg();
//            this.oi.viewOrg();
        }
        if (network.RUN_MODEL == RunModel.EVAL) {
            // [batch_size，time，head_num，d_k]
            this.qt = CUDAMemoryManager.getCache("attn-qt", batchSize, time, 1, channel);
//            this.kt = CUDAMemoryManager.getCache("attn-kt", batchSize, time, 1, channel);
//            this.vt = CUDAMemoryManager.getCache("attn-vt", batchSize, time, 1, channel);
            // [batch_size，n_heads，len_q，len_k]
            if (time < channel) {
                this.temp = CUDAMemoryManager.getCache("attn-temp1", batchSize, 1, time, channel);
            } else {
                this.temp = CUDAMemoryManager.getCache("attn-temp2", batchSize, 1, time, time);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = CUDAMemoryManager.getCache("attn-attn", batchSize, 1, time, time);
            // [batch_size, len_q, n_heads * dim_v]
//            this.oi = CUDAMemoryManager.getCache("attn-oi", batchSize * time, 1, 1, embedDim);
        } else {
            if (this.qt == null || this.qt.number != this.batchSize) {
                // [batch_size，time，head_num，d_k]
                this.qt = Tensor.createGPUTensor(this.qt, batchSize, time, 1, channel, true);
//                this.kt = Tensor.createGPUTensor(this.kt, batchSize, time, 1, channel, true);
//                this.vt = Tensor.createGPUTensor(this.vt, batchSize, time, 1, channel, true);
                // [batch_size，n_heads，len_q，len_k]
                if (time < channel) {
                    this.temp = Tensor.createGPUTensor(this.temp, batchSize, 1, time, channel, true);
                } else {
                    this.temp = Tensor.createGPUTensor(this.temp, batchSize, 1, time, time, true);
                }
                // [batch_size，n_heads，len_q，len_k]
                this.attn = Tensor.createGPUTensor(this.attn, batchSize, 1, time, time, true);
                // [batch_size, len_q, n_heads * dim_v]
//                this.oi = Tensor.createGPUTensor(this.oi, batchSize, time, 1, channel, true);
            }
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
        	if (network.gradCacheMode) {
        		if(this.dqkvt == null || !this.dqkvt.checkShape(qt)) {
        			this.dqkvt = network.cudaManager.getMemoryManager().getPrivateCaches("attn-dqkvt", batchSize, 1, time, channel);
                    this.dattn = network.cudaManager.getMemoryManager().getPrivateCaches("attn-dattn", batchSize, 1, time, time);
        		}
        	}else {
        		this.dqkvt = Tensor.createGPUTensor(this.dqkvt, batchSize, 1, time, channel, true);
                this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, 1, time, time, true);
        	}
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        if (network.RUN_MODEL == RunModel.EVAL) {
            eval();
        } else {
            train();
        }
    }

    public void eval() {
        Tensor qfo = CUDAMemoryManager.getCache("VQVAEAttn_qfo_cache", batchSize, channel, height, width);
        Tensor kfo = CUDAMemoryManager.getCache("VQVAEAttn_kfo_cache", batchSize, channel, height, width);
        Tensor vfo = CUDAMemoryManager.getCache("VQVAEAttn_vfo_cache", batchSize, channel, height, width);
        this.qLinerLayer.forward(this.input, qfo);
        this.kLinerLayer.forward(this.input, kfo);
        this.vLinerLayer.forward(this.input, vfo);
        Tensor q = this.qLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor k = this.kLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor v = this.vLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor_OP().permute(q, qt, new int[]{0, 3, 2, 1});//batchSize, time, 1, channel
        scaledDotProductAttention(qt, k, v);
        Tensor vaccum = temp;
//        attentionKernel.unpermute(vaccum, oi, batchSize, channel, 1, time);
        oLinerLayer.forward(vaccum.view(batchSize, channel, height, width));
        this.output = oLinerLayer.getOutput();
        vaccum.viewOrg();
    }

    //	public void scaledDotProductAttentionBackward() {
    //
    //		Tensor dvaccum = temp;
    //	    // backward into datt
    //		GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
    //
    //		// backward into dv
    //		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, batchSize * headNum);
    //
    //		// backward into preatt
    //		softmaxKernel.softmax_backward(attn, dattn, dattn);
    //		float d_k = (float) (1.0f / Math.sqrt(dk));
    //		Tensor_OP().mul(dattn, d_k, dattn);
    //		Tensor dpreatt = dattn;
    //
    //		// backward into q
    //		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
    //
    //		// backward into k
    //		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
    //	}
    
    public void train() {
        this.qLinerLayer.forward(this.input);
        this.kLinerLayer.forward(this.input);
        this.vLinerLayer.forward(this.input);
        /**
         * b,c,hw =>b,hw,c
         */
        Tensor q = this.qLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor k = this.kLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor v = this.vLinerLayer.getOutput().view(batchSize, channel, 1, time);
        Tensor_OP().permute(q, qt, new int[]{0, 3, 2, 1});
        scaledDotProductAttention(qt, k, v);
        Tensor vaccum = temp;
//        attentionKernel.unpermute(vaccum, oi, batchSize, channel, 1, time);
        oLinerLayer.forward(vaccum);
        this.output = oLinerLayer.getOutput();
    }

//    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
//        float d_k = (float) (1.0f / Math.sqrt(channel));
//        Tensor preatt = temp;
//        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, channel, time, channel, 1.0f, key.getGpuData(), channel, time * channel, query.getGpuData(), channel, time * channel, 0.0f, preatt.getGpuData(), time, time * time, batchSize * 1);
//        Tensor_OP().mul(preatt, d_k, preatt);
//        softmaxKernel.softmax(preatt, attn, batchSize * 1 * time);
//        preatt.showDMByOffsetRed(0, 100, "preatt");
//        Tensor tmp = attn;
//        Tensor vaccum = temp;
//        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, channel, time, time, 1.0f, value.getGpuData(), channel, time * channel, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), channel, time * channel, batchSize * 1);
//    }
    
    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(channel));
        Tensor preatt = temp;
        GPU_OP().bmm(query.getGpuData(), key.getGpuData(), preatt.getGpuData(), batchSize, time, time, channel, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
        Tensor_OP().mul(preatt, d_k, preatt);
        softmaxKernel.softmax(preatt, attn, batchSize * time);
//        attn.showDMByOffsetRed(0, time, "attn");
        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPU_OP().bmm(value.getGpuData(), tmp.getGpuData(), vaccum.getGpuData(), batchSize, channel, time, time, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);
//        vaccum.showDMByOffsetRed(0, time, "vaccum");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
//        oLinerLayer.back(delta, oi);
//        attentionKernel.unpermute_backward(temp, oi, batchSize, time, 1, channel);
//        Tensor dvaccum = temp;
//        // backward into datt
//        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, channel, 1.0f, vt.getGpuData(), channel, time * channel, dvaccum.getGpuData(), channel, time * channel, 0.0f, dattn.getGpuData(), time, time * time, batchSize * 1);
//        // backward into preatt
//        softmaxKernel.softmax_backward(attn, dattn, dattn);
//        float d_k = (float) (1.0f / Math.sqrt(channel));
//        Tensor_OP().mul(dattn, d_k, dattn);
//        Tensor dpreatt = dattn;
//        // backward into dv
//        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, channel, time, time, 1.0f, dvaccum.getGpuData(), channel, time * channel, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), channel, time * channel, batchSize * 1);
//        Tensor dvt = vt.view(batchSize, time, 1, channel);
//        Tensor_OP().permute(dqkvt, dvt, new int[]{0, 2, 1, 3});
//        Tensor vDelta = dvt.view(batchSize * time, 1, 1, 1 * channel);
//        this.vLinerLayer.back(vDelta, output);
//        Tensor_OP().add(output, 0, delta);
//        // backward into q
//        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, channel, time, time, 1.0f, kt.getGpuData(), channel, time * channel, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), channel, time * channel, batchSize * 1);
//        Tensor dqt = vt.view(batchSize, time, 1, channel);
//        Tensor_OP().permute(dqkvt, dqt, new int[]{0, 2, 1, 3});
//        Tensor qDelta = dqt.view(batchSize * time, 1, 1, 1 * channel);
//        this.qLinerLayer.back(qDelta, output);
//        Tensor_OP().add(delta, output, delta);
//        // backward into k
//        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, channel, time, time, 1.0f, qt.getGpuData(), channel, time * channel, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), channel, time * channel, batchSize * 1);
//        Tensor dkt = vt.view(batchSize, time, 1, channel);
//        Tensor_OP().permute(dqkvt, dkt, new int[]{0, 2, 1, 3});
//        Tensor kDelta = dkt.view(batchSize * time, 1, 1, 1 * channel);
//        this.kLinerLayer.back(kDelta, output);
//        Tensor_OP().add(delta, output, delta);
////        Tensor_OP().add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
////        Tensor_OP().add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);
//        this.diff = delta;
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
        this.setInput(input);
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
        oLinerLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        qLinerLayer.loadModel(inputStream);
        kLinerLayer.loadModel(inputStream);
        vLinerLayer.loadModel(inputStream);
        oLinerLayer.loadModel(inputStream);
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

