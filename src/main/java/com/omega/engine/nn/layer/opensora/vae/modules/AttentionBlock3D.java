package com.omega.engine.nn.layer.opensora.vae.modules;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * AttentionBlock3D
 *
 * @author Administrator
 */
public class AttentionBlock3D extends Layer {
	
	public GNLayer3D norm;
    public CausalConv3DPlainAR qLinerLayer;
    public CausalConv3DPlainAR kLinerLayer;
    public CausalConv3DPlainAR vLinerLayer;
    public CausalConv3DPlainAR oLinerLayer;
    
    public int depth = 0;;
    public int oDepth = 0;
    
    private boolean bias = false;
    
    private boolean isCausal = true;
    
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    
    private int time;
    private int headNum = 1;
    private int dk;

    private float maskFill = -1e9f;
    
    private Tensor mask;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    private Tensor dqkvt;
    private Tensor d_tmp;
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    private Tensor oit;
    private Tensor dattn;

    public AttentionBlock3D(int channel, int depth, int height, int width, boolean isCausal, boolean bias, Network network) {
        this.bias = bias;
        this.isCausal = isCausal;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.bias = bias;
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
        this.time = depth * height * width;
        this.dk = channel;
        this.initLayers();
    }

    public static void main(String[] args) {
    	int batchSize = 2;
    	int channel = 128;
    	int numFrames = 5;
    	int imageSize = 4;
    	Tensor input = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, true);
    	
        Transformer tf = new Transformer();
        tf.CUDNN = true;
        float[] data = RandomUtils.order(input.dataLength, 0.001f, 0.001f);
        Tensor input2 = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, data, true);
        
//        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, data, true);
        AttentionBlock3D mal = new AttentionBlock3D(channel, numFrames, imageSize, imageSize, true, true, tf);
        
        String weight = "D:\\models\\opensora_attn3d.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), mal, true);
        
        for (int i = 0; i < 2; i++) {
            mal.forward(input2);
            mal.getOutput().showShape();
            mal.getOutput().showDM();
            mal.back(delta);
            mal.diff.showDM();
        }
    }
    
    public static void loadWeight(Map<String, Object> weightMap, AttentionBlock3D network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        network.norm.norm.gamma = ClipModelUtils.loadData(network.norm.norm.gamma, weightMap, 1, "norm.weight");
        network.norm.norm.beta = ClipModelUtils.loadData(network.norm.norm.beta, weightMap, 1, "norm.bias");
        
        ClipModelUtils.loadData(network.qLinerLayer.weight, weightMap, "q.conv.weight", 5);
        ClipModelUtils.loadData(network.qLinerLayer.bias, weightMap, "q.conv.bias");
        ClipModelUtils.loadData(network.kLinerLayer.weight, weightMap, "k.conv.weight", 5);
        ClipModelUtils.loadData(network.kLinerLayer.bias, weightMap, "k.conv.bias");
        ClipModelUtils.loadData(network.vLinerLayer.weight, weightMap, "v.conv.weight", 5);
        ClipModelUtils.loadData(network.vLinerLayer.bias, weightMap, "v.conv.bias");
        ClipModelUtils.loadData(network.oLinerLayer.weight, weightMap, "proj_out.conv.weight", 5);
        ClipModelUtils.loadData(network.oLinerLayer.bias, weightMap, "proj_out.conv.bias");
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
    	
    	this.norm = new GNLayer3D(channel, depth, height, width, 32, network);
    	
        this.qLinerLayer = new CausalConv3DPlainAR(channel, channel, depth, width, height, 1, 1, bias, network);
        this.kLinerLayer = new CausalConv3DPlainAR(channel, channel, depth, width, height, 1, 1, bias, network);
        this.vLinerLayer = new CausalConv3DPlainAR(channel, channel, depth, width, height, 1, 1, bias, network);
        this.oLinerLayer = new CausalConv3DPlainAR(channel, channel, depth, width, height, 1, 1, bias, network);

        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
        }
        
        if(isCausal) {
        	mask = new Tensor(1, 1, time, time, MatrixUtils.triu(1, 1, time, time, 1), true);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        
        if (this.qt != null) {
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
            this.oi.viewOrg();
            this.oit.viewOrg();
            this.qLinerLayer.getOutput().viewOrg();
            this.kLinerLayer.getOutput().viewOrg();
            this.vLinerLayer.getOutput().viewOrg();
        }
        if (this.qt == null || this.qt.number != this.number) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createGPUTensor(this.qt, number, headNum, time, dk, true);
            this.kt = Tensor.createGPUTensor(this.kt, number, headNum, time, dk, true);
            this.vt = Tensor.createGPUTensor(this.vt, number, headNum, time, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            if (time < dk) {
                this.temp = Tensor.createGPUTensor(this.temp, number, headNum, time, dk, true);
            } else {
                this.temp = Tensor.createGPUTensor(this.temp, number, headNum, time, time, true);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createGPUTensor(this.attn, number, headNum, time, time, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, number, time, 1, channel, true);
            this.oit = Tensor.createGPUTensor(this.oit, number, channel, 1, time, true);
        }
        if(this.output == null || this.output.number != number) {
        	this.output = Tensor.createGPUTensor(output, number, channel * depth, height, width, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
            this.dqkvt = Tensor.createGPUTensor(this.dqkvt, number, headNum, time, dk, true);
            this.dattn = Tensor.createGPUTensor(this.dattn, number, headNum, time, time, true);
            this.d_tmp = Tensor.createGPUTensor(this.d_tmp, input.shape(), true);
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
//            eval();
        } else {
            train();
        }
    }

//    public void eval() {
//        Tensor qfo = CUDAMemoryManager.getCache("VQVAEAttn_qfo_cache", batchSize * time, 1, 1, embedDim);
//        Tensor kfo = CUDAMemoryManager.getCache("VQVAEAttn_kfo_cache", batchSize * time, 1, 1, embedDim);
//        Tensor vfo = CUDAMemoryManager.getCache("VQVAEAttn_vfo_cache", batchSize * time, 1, 1, embedDim);
//        this.qLinerLayer.forward(this.input, qfo);
//        this.kLinerLayer.forward(this.input, kfo);
//        this.vLinerLayer.forward(this.input, vfo);
//        Tensor q = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
//        Tensor k = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
//        Tensor v = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
//        Tensor_OP().permute(q, qt, new int[]{0, 2, 1, 3});
//        Tensor_OP().permute(k, kt, new int[]{0, 2, 1, 3});
//        Tensor_OP().permute(v, vt, new int[]{0, 2, 1, 3});
//        scaledDotProductAttention(qt, kt, vt);
//        Tensor vaccum = temp;
//        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
//        this.getoLinerLayer().forward(oi);
//        this.output = this.getoLinerLayer().getOutput();
//    }

    public void train() {

    	this.norm.forward(input);
//    	norm.getOutput().showDM("norm");
        this.qLinerLayer.forward(norm.getOutput());
        this.kLinerLayer.forward(norm.getOutput());
        this.vLinerLayer.forward(norm.getOutput());
        
        Tensor q = this.qLinerLayer.getOutput().view(number, channel, 1, time);
        Tensor k = this.kLinerLayer.getOutput().view(number, channel, 1, time);
        Tensor v = this.vLinerLayer.getOutput().view(number, channel, 1, time);

        Tensor_OP().permute(q, qt, new int[]{0, 2, 3, 1});
        Tensor_OP().permute(k, kt, new int[]{0, 2, 3, 1});
        Tensor_OP().permute(v, vt, new int[]{0, 2, 3, 1});

        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, number, time, headNum, dk);

        Tensor_OP().permute(oi, oit, new int[]{0, 3, 2, 1});  //n t 1 c -> n c 1 t
        oit.view(number, channel * depth , height, width);
        this.oLinerLayer.forward(oit);
        
        Tensor_OP().add(this.oLinerLayer.getOutput(), input, this.output);

    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, number * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);

        if(isCausal) {
        	Tensor_OP().mask(preatt, mask, preatt, maskFill, number * mask.dataLength, mask.dataLength);
        }

        softmaxKernel.softmax(preatt, attn, number * headNum * time);

        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, number * headNum);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.oLinerLayer.back(delta, oit);
        oit.viewOrg();
//        oit.showDM("oit");
        Tensor_OP().permute(oit, oi, new int[]{0, 3, 2, 1});  //n c 1 t -> n t 1 c
        attentionKernel.unpermute_backward(temp, oi, number, time, headNum, dk);
        Tensor dvaccum = temp;
        // backward into datt
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, number * headNum);
//        dattn.showDMByOffsetRed(79 * 80, 80, "dattn");
        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        // backward into dv
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum);
        Tensor dvt = vt.view(number, channel, 1, time); // b, c, 1, t
        Tensor_OP().permute(dqkvt, dvt, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        Tensor vDelta = dvt.view(this.vLinerLayer.getOutput().shape());
        this.vLinerLayer.back(vDelta, norm.getOutput());
        Tensor_OP().add(norm.getOutput(), 0, d_tmp);
        // backward into q
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum);
        Tensor dqt = vt.view(number, channel, 1, time); // b, c, 1, t
        Tensor_OP().permute(dqkvt, dqt, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        Tensor qDelta = dvt.view(this.qLinerLayer.getOutput().shape());
        this.qLinerLayer.back(qDelta, norm.getOutput());

        Tensor_OP().add(norm.getOutput(), d_tmp, d_tmp);
        // backward into k
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum);
        Tensor dkt = vt.view(number, channel, 1, time); // b, c, 1, t
        Tensor_OP().permute(dqkvt, dkt, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        Tensor kDelta = dvt.view(this.kLinerLayer.getOutput().shape());
        this.kLinerLayer.back(kDelta, norm.getOutput());

        Tensor_OP().add(norm.getOutput(), d_tmp, d_tmp);
        this.norm.back(d_tmp);
        Tensor_OP().add(this.norm.diff, delta, this.norm.diff);
        this.diff = this.norm.diff;
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
    	norm.update();
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
    	norm.saveModel(outputStream);
        qLinerLayer.saveModel(outputStream);
        kLinerLayer.saveModel(outputStream);
        vLinerLayer.saveModel(outputStream);
        oLinerLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream);
        qLinerLayer.loadModel(inputStream);
        kLinerLayer.loadModel(inputStream);
        vLinerLayer.loadModel(inputStream);
        oLinerLayer.loadModel(inputStream);
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

