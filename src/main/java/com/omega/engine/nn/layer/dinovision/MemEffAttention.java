package com.omega.engine.nn.layer.dinovision;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * MemEffAttention
 *
 * @author Administrator
 */
public class MemEffAttention extends Layer {
	
    public FullyLayer qkvLinerLayer;
    public FullyLayer oLinerLayer;
    
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;

    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    
    private Tensor tmp;

    private Tensor qt;
    private Tensor kt;
    private Tensor vt;

    private Tensor temp;
    private Tensor attn;
    private Tensor oi;

    private int batchSize = 1;
    
    private int[] p_0213 = new int[]{0, 2, 1, 3};
    

    public MemEffAttention(int embedDim, int headNum, int time, boolean bias) {
        this.bias = bias;
        this.time = time;
        this.embedDim = embedDim;
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
        this.initLayers();
    }

    public MemEffAttention(int embedDim, int headNum, int time, boolean bias, Network network) {
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
        this.channel = time;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.initLayers();
    }
    public void initLayers() {
       
        this.qkvLinerLayer = new FullyLayer(embedDim, embedDim * 3, bias, this.network);

        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        
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

    public void init_eval(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number/time;
        this.tmp = CUDAMemoryManager.getCache("dit_block_attn_tmp", batchSize * time, 1, 1, embedDim);
    	this.qt = CUDAMemoryManager.getCache("dit_block_attn_qt", batchSize, headNum, time, dk);
    	this.kt = CUDAMemoryManager.getCache("dit_block_attn_kt", batchSize, headNum, time, dk);
    	this.vt = CUDAMemoryManager.getCache("dit_block_attn_vt", batchSize, headNum, time, dk);
        // [batch_size，n_heads，len_q，len_k]
        if (time < dk) {
            this.temp = CUDAMemoryManager.getCache("dit_block_attn_temp", batchSize, headNum, time, dk);
        } else {
            this.temp = CUDAMemoryManager.getCache("dit_block_attn_temp", batchSize, time, time, dk);
        }
        temp.clearGPU();
        // [batch_size，n_heads，len_q，len_k]
        this.attn = CUDAMemoryManager.getCache("dit_block_attn_attn", batchSize, time, time, dk);
        // [batch_size, len_q, n_heads * dim_v]
        this.oi = CUDAMemoryManager.getCache("dit_block_attn_oi", batchSize * time, 1, 1, embedDim);
        this.output = CUDAMemoryManager.getCache("dit_block_attn_out", input.number, input.channel, input.height, input.width);
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
       
    }
    
    public void output_eval() {
         // TODO Auto-generated method stub
    	 this.getqkvLinerLayer().forward(input);

    	 Tensor_OP().getByChannel(getqkvLinerLayer().getOutput(), tmp, new int[] {input.number, 3, 1, embedDim}, 0);
         Tensor query = tmp.view(batchSize, time, headNum, dk);
         Tensor_OP().permute(query, qt, p_0213);
         
         Tensor_OP().getByChannel(getqkvLinerLayer().getOutput(), tmp, new int[] {input.number, 3, 1, embedDim}, 1);
         Tensor key = tmp.view(batchSize, time, headNum, dk);
         Tensor_OP().permute(key, kt, p_0213);
         
         Tensor_OP().getByChannel(getqkvLinerLayer().getOutput(), tmp, new int[] {input.number, 3, 1, embedDim}, 2);
         Tensor value = tmp.view(batchSize, time, headNum, dk);
         Tensor_OP().permute(value, vt, p_0213);

         scaledDotProductAttention(qt, kt, vt);
         Tensor vaccum = temp;
         attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

         this.getoLinerLayer().forward(oi, output);
         this.output = this.getoLinerLayer().getOutput();
    }
    
    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);
        softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        Tensor tmp = attn;

        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        
    }
    
    public void diff(Tensor cos, Tensor sin) {
    	 
    }
    
    public void diff(Tensor cos, Tensor sin, int igone) {
    	
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
        this.init_eval(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output_eval();
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
        getqkvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        getqkvLinerLayer().loadModel(inputStream);
        getoLinerLayer().loadModel(inputStream);
    }

    public FullyLayer getqkvLinerLayer() {
        return qkvLinerLayer;
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

    }

}

