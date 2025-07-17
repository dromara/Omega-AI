package com.omega.engine.nn.layer.opensora.wfvae.modules;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.opensora.vae.modules.GNLayer3D;
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
public class WFAttentionBlock3D extends Layer {
	
	public GNLayer3D norm;
    public WFCausalConv3D qLinerLayer;
    public WFCausalConv3D kLinerLayer;
    public WFCausalConv3D vLinerLayer;
    public WFCausalConv3D oLinerLayer;
    
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
    private Tensor dattn;

    public WFAttentionBlock3D(int channel, int depth, int height, int width, boolean isCausal, boolean bias, Network network) {
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
        this.time = height * width;
        this.dk = channel;
        this.initLayers();
    }

    public static void main(String[] args) {
    	
    	 int N = 2;
         int C = 64;
         int F = 5;
         int H = 32;
         int W = 32;

         String inputPath = "D:\\models\\input_wf.json";
         Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
         Tensor input = new Tensor(N, C * F, H, W, true);
         ClipModelUtils.loadData(input, datas, "x", 5);
    	
        Transformer tf = new Transformer();
        tf.CUDNN = true;

        WFAttentionBlock3D mal = new WFAttentionBlock3D(C, F, H, W, false, false, tf);
        
        String weight = "D:\\models\\wf_attn3d.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), mal, true);
        
        String deltaPath = "D:\\models\\attn3d_delta_wf.json";
        Map<String, Object> d = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(delta, d, "delta", 5);
        
        for (int i = 0; i < 1; i++) {
            mal.forward(input);
//            mal.getOutput().showShape();
//            mal.getOutput().showDM();
            mal.back(delta);
//            mal.diff.showDM();
        }
      mal.getOutput().showShape();
      mal.getOutput().showDM();
      mal.diff.showDM();
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFAttentionBlock3D network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        network.norm.norm.gamma = ClipModelUtils.loadData(network.norm.norm.gamma, weightMap, 1, "norm.weight");
        network.norm.norm.beta = ClipModelUtils.loadData(network.norm.norm.beta, weightMap, 1, "norm.bias");
        
        ClipModelUtils.loadData(network.qLinerLayer.weight, weightMap, "q.conv.weight", 5);
//        ClipModelUtils.loadData(network.qLinerLayer.bias, weightMap, "q.conv.bias");
        ClipModelUtils.loadData(network.kLinerLayer.weight, weightMap, "k.conv.weight", 5);
//        ClipModelUtils.loadData(network.kLinerLayer.bias, weightMap, "k.conv.bias");
        ClipModelUtils.loadData(network.vLinerLayer.weight, weightMap, "v.conv.weight", 5);
//        ClipModelUtils.loadData(network.vLinerLayer.bias, weightMap, "v.conv.bias");
        ClipModelUtils.loadData(network.oLinerLayer.weight, weightMap, "proj_out.conv.weight", 5);
//        ClipModelUtils.loadData(network.oLinerLayer.bias, weightMap, "proj_out.conv.bias");
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
    	
        this.qLinerLayer = new WFCausalConv3D(channel, channel, depth, width, height, 1, 1, bias, network);
        this.kLinerLayer = new WFCausalConv3D(channel, channel, depth, width, height, 1, 1, bias, network);
        this.vLinerLayer = new WFCausalConv3D(channel, channel, depth, width, height, 1, 1, bias, network);
        this.oLinerLayer = new WFCausalConv3D(channel, channel, depth, width, height, 1, 1, bias, network);

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
            this.qLinerLayer.getOutput().viewOrg();
            this.kLinerLayer.getOutput().viewOrg();
            this.vLinerLayer.getOutput().viewOrg();
        }
        if (this.qt == null || this.qt.number != this.number) {
            // [batch_size，time，head_num，d_k]
            this.qt = Tensor.createGPUTensor(this.qt, number * depth, headNum, time, dk, true);
            this.kt = Tensor.createGPUTensor(this.kt, number * depth, headNum, time, dk, true);
            this.vt = Tensor.createGPUTensor(this.vt, number * depth, headNum, time, dk, true);
            // [batch_size，n_heads，len_q，len_k]
            if (time < dk) {
                this.temp = Tensor.createGPUTensor(this.temp, number * depth, headNum, time, dk, true);
            } else {
                this.temp = Tensor.createGPUTensor(this.temp, number * depth, headNum, time, time, true);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createGPUTensor(this.attn, number * depth, headNum, time, time, true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, number * depth, time, 1, channel, true);
        }
        if(this.output == null || this.output.number != number) {
        	this.output = Tensor.createGPUTensor(output, number, channel * depth, height, width, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
            this.dqkvt = Tensor.createGPUTensor(this.dqkvt, number * depth, headNum, time, dk, true);
            this.dattn = Tensor.createGPUTensor(this.dattn, number * depth, headNum, time, time, true);
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

        Tensor q = this.qLinerLayer.getOutput().view(number, channel, depth, time);
        Tensor k = this.kLinerLayer.getOutput().view(number, channel, depth, time);
        Tensor v = this.vLinerLayer.getOutput().view(number, channel, depth, time);

        int[] a_shape = new int[] {number, channel, depth, time};
        int[] b_shape = new int[] {number, depth, time, channel};
        
        Tensor_OP().permute(q, qt, a_shape, b_shape, new int[]{0, 2, 3, 1});
        Tensor_OP().permute(k, kt, a_shape, b_shape, new int[]{0, 2, 3, 1});
        Tensor_OP().permute(v, vt, a_shape, b_shape, new int[]{0, 2, 3, 1});
       
        scaledDotProductAttention(qt, kt, vt);
        Tensor vaccum = temp;
        
        Tensor_OP().permute(vaccum, oi, new int[] {number, depth, time, channel}, new int[] {number, channel, depth, time}, new int[] {0, 3, 1, 2}, oi.dataLength);
        
        this.oLinerLayer.forward(oi);
        
        Tensor_OP().add(this.oLinerLayer.getOutput(), input, this.output);

    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, number * headNum * depth);
        Tensor_OP().mul(preatt, d_k, preatt);
 
        if(isCausal) {
        	Tensor_OP().mask(preatt, mask, preatt, maskFill, number * depth * mask.dataLength, mask.dataLength);
        }

        softmaxKernel.softmax(preatt, attn, number * depth * headNum * time);
        
        Tensor tmp = attn;
        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, number * headNum * depth);

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.oLinerLayer.back(delta, oi);
        Tensor_OP().permute(oi, temp, new int[] {number, channel, depth, time}, new int[] {number, depth, time, channel}, new int[]{0, 2, 3, 1}, oi.dataLength);  //n c 1 t -> n t 1 c
        Tensor dvaccum = temp;
        // backward into datt
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, number * headNum * depth);
//        dattn.showDMByOffsetRed(79 * 80, 80, "dattn");
        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        // backward into dv
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum * depth);
        Tensor_OP().permute(dqkvt, this.vLinerLayer.getOutput(), new int[] {number, depth, time, channel}, new int[] {number, channel, depth, time}, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        this.vLinerLayer.back(this.vLinerLayer.getOutput(), norm.getOutput());
        Tensor_OP().add(norm.getOutput(), 0, d_tmp);
        // backward into q
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum * depth);
        Tensor_OP().permute(dqkvt, this.qLinerLayer.getOutput(), new int[] {number, depth, time, channel}, new int[] {number, channel, depth, time}, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        this.qLinerLayer.back(this.qLinerLayer.getOutput(), norm.getOutput());
        Tensor_OP().add(norm.getOutput(), d_tmp, d_tmp);
        // backward into k
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, number * headNum * depth);
        Tensor_OP().permute(dqkvt, this.kLinerLayer.getOutput(), new int[] {number, depth, time, channel}, new int[] {number, channel, depth, time}, new int[]{0, 3, 1, 2});//n 1 t c-> n c 1 t
        this.kLinerLayer.back(kLinerLayer.getOutput(), norm.getOutput());
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

