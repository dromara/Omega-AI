package com.omega.engine.nn.layer.dit.mmdit;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiTAttentionLayer2
 *
 * @author Administrator
 */
public class DiTJoinBlock extends Layer {
    
    private boolean qkNorm = false;
    
    private int batchSize = 1;
    
    private int mlp_ratio;
    private int time;
    public int imgTime;
    private int textTime;
    private int headNum = 1;
    private int embedDim = 0;
    private int cEmbedDim = 0;
    private int dk = 0;

    private boolean bias = false;
    private boolean normParams = true;
    private boolean pre_only = false;
    
    public DiTJoinBlockHead x_block;
    public DiTJoinBlockHead context_block;
    
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;

    private Tensor q;
    private Tensor k;
    private Tensor v;
    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
   
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    
    private Tensor x_attn;
    private Tensor context_attn;
    
    private Tensor cx_diff;
    private Tensor dattn_cx;
    private Tensor dattn;
    
    private int[] shape;
    private int[] t_shape;

    public DiTJoinBlock(int embedDim, int cEmbedDim, int mlp_ratio, int headNum, int imgTime, int textTime, boolean bias, boolean qkNorm, boolean pre_only, boolean normParams, Network network) {
        this.bias = bias;
        this.mlp_ratio = mlp_ratio;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.imgTime = imgTime;
        this.textTime = textTime;
        this.time = imgTime + textTime;
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.qkNorm = qkNorm;
        this.normParams = normParams;
        this.pre_only = pre_only;
        this.dk = embedDim / headNum;
        this.bias = bias;
        this.channel = imgTime;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.initLayers();
    }
    
    public void initLayers() {
       
    	x_block = new DiTJoinBlockHead(embedDim, cEmbedDim, mlp_ratio, imgTime, bias, qkNorm, false, normParams, network);
    	
    	context_block = new DiTJoinBlockHead(embedDim, cEmbedDim, mlp_ratio, textTime, bias, qkNorm, pre_only, normParams, network);
    	
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
        this.batchSize = this.number / imgTime;
        if (this.qt != null) {
            this.output.viewOrg();
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
        }
        if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
        	shape = new int[] {batchSize, time, headNum, dk};
        	t_shape = new int[] {batchSize, headNum, time, dk};
            // [batch_size，time，head_num，d_k]
            this.q = Tensor.createGPUTensor(this.q, batchSize, time, 1, embedDim, true);
            this.k = Tensor.createGPUTensor(this.k, batchSize, time, 1, embedDim, true);
            this.v = Tensor.createGPUTensor(this.v, batchSize, time, 1, embedDim, true);
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
            this.oi = Tensor.createGPUTensor(this.oi, batchSize, time, 1, embedDim, true);
            this.x_attn = Tensor.createGPUTensor(this.x_attn, batchSize * imgTime, 1, 1, embedDim, true);
            this.context_attn = Tensor.createGPUTensor(this.context_attn, batchSize * textTime, 1, 1, embedDim, true);
        }
    }
    
    public void init_eval(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number / imgTime;
    	this.qt = CUDAMemoryManager.getCache("dit_block_attn_qt", batchSize, headNum, time, dk);
    	this.kt = CUDAMemoryManager.getCache("dit_block_attn_kt", batchSize, headNum, time, dk);
    	this.vt = CUDAMemoryManager.getCache("dit_block_attn_vt", batchSize, headNum, time, dk);
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null || dattn.number != batchSize) {
//            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
            this.dattn = CUDAMemoryManager.getCache("dit_block_dattn", batchSize, headNum, time, time);
            if(pre_only) {
//            	dattn_cx = Tensor.createGPUTensor(this.dattn_cx, batchSize, textTime, 1, embedDim, true);
//            	cx_diff = Tensor.createGPUTensor(this.cx_diff, batchSize, textTime, 1, embedDim, true);
            	this.dattn_cx = CUDAMemoryManager.getCache("dit_block_dattn_cx", batchSize, textTime, 1, embedDim);
            	this.cx_diff = CUDAMemoryManager.getCache("dit_block_cx_diff", batchSize, textTime, 1, embedDim);
            }
        } else {
            dattn.viewOrg();
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
    
    public void output(Tensor context, Tensor c) {
    	
    	x_block.pre_attention(input, c); //[batchSize, imgTime, embedDim]
    	context_block.pre_attention(context, c);  //[batchSize, textTime, embedDim]
    	
    	attentionKernel.concat_channel_forward(context_block.q(), x_block.q(), q, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(context_block.k(), x_block.k(), k, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(context_block.v(), x_block.v(), v, batchSize, textTime, imgTime, 1, embedDim);
    	
    	Tensor query = q.view(batchSize, time, headNum, dk);
        Tensor key = k.view(batchSize, time, headNum, dk);
        Tensor value = v.view(batchSize, time, headNum, dk);
    	
        Tensor_OP().permute(query, qt, shape, t_shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, shape, t_shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, shape, t_shape, new int[]{0, 2, 1, 3});
        
        scaledDotProductAttention(qt, kt, vt);
        
        attentionKernel.unpermute(temp, oi, batchSize, time, headNum, dk);
        attentionKernel.concat_channel_backward(oi, context_attn, x_attn, batchSize, textTime, imgTime, 1, embedDim);
        
        x_block.post_attention(x_attn);
        if(!pre_only) {
        	context_block.post_attention(context_attn);
        }

        this.output = x_block.getOutput();
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
    
    public void scaledDotProductAttentionBackward() {
        Tensor tmp = attn;

        Tensor dvaccum = temp;
        /**
         * backward into dattn[b, nh, t, t2]
         * vt[b, nh, t2, dk] -> [b, nh, dk, t2]
         * dvaccum[b, nh, t, dk]
         */
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        /**
         * backward into dvt[b, nh, t2, dk]
         * dvaccum[b, nh, t, dk]
         * attn[b, nh, t, t2] -> [b, nh, t2, t]
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, v.getGpuData(), dk, time * dk, batchSize * headNum);

        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        //		dattn.showDM();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        /**
         * backward into dqt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, q.getGpuData(), dk, time * dk, batchSize * headNum);
        /**
         * backward into dkt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, k.getGpuData(), dk, time * dk, batchSize * headNum);
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
    
    public void diff(Tensor dcx,Tensor dtc) {
        // TODO Auto-generated method stub
//    	delta.showDM("x");
//    	dcx.showDM("dcx");
        Tensor x_diff = x_block.post_attention_back(delta, "x");
        Tensor dattn = x_block.oLinerLayer.diff;
        if(!pre_only) {
        	cx_diff = context_block.post_attention_back(dcx, "cx");
        	dattn_cx = context_block.oLinerLayer.diff;
        }
        
        attentionKernel.concat_channel_forward(dattn_cx, dattn, oi, batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        
        scaledDotProductAttentionBackward();
        
        Tensor_OP().permute(q, qt, t_shape, shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(k, kt, t_shape, shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(v, vt, t_shape, shape, new int[]{0, 2, 1, 3});
        
        attentionKernel.concat_channel_backward(qt, context_block.q(), x_block.q(), batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.concat_channel_backward(kt, context_block.k(), x_block.k(), batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.concat_channel_backward(vt, context_block.v(), x_block.v(), batchSize, textTime, imgTime, 1, embedDim);
        
        x_block.pre_attention_back(x_diff, dtc);
//        x_block.diff.showDMByOffsetRed(0, 10, "x_diff");
        context_block.pre_attention_back(cx_diff, dtc);
//        context_block.diff.showDMByOffsetRed(0, 10, "cx_diff");
        this.diff = x_block.diff;
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
    
    public void forward(Tensor input, Tensor context, Tensor c) {
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
        this.output(context, c);
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
    
    public void back(Tensor delta, Tensor cx_delta, Tensor ctd) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cx_delta, ctd);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	x_block.update();
    	context_block.update();
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
    	x_block.saveModel(outputStream);
    	context_block.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	x_block.loadModel(inputStream);
    	context_block.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }
    
    public static void loadWeight(Map<String, Object> weightMap, DiTJoinBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        block.context_block.norm1.gamma = ModeLoaderlUtils.loadData(block.context_block.norm1.gamma, weightMap, 1, "context_block.norm1.weight"); 
        ModeLoaderlUtils.loadData(block.context_block.qLinerLayer.weight, weightMap, "context_block.attn.ql.weight");
        ModeLoaderlUtils.loadData(block.context_block.kLinerLayer.weight, weightMap, "context_block.attn.kl.weight");
        ModeLoaderlUtils.loadData(block.context_block.vLinerLayer.weight, weightMap, "context_block.attn.vl.weight");
        if(block.context_block.oLinerLayer != null) {
        	 ModeLoaderlUtils.loadData(block.context_block.oLinerLayer.weight, weightMap, "context_block.attn.proj.weight");
             ModeLoaderlUtils.loadData(block.context_block.oLinerLayer.bias, weightMap, "context_block.attn.proj.bias");
        }
        block.context_block.norm2.gamma = ModeLoaderlUtils.loadData(block.context_block.norm2.gamma, weightMap, 1, "context_block.norm2.weight"); 
        if(block.context_block.mlp != null) {
        	 ModeLoaderlUtils.loadData(block.context_block.mlp.linear1.weight, weightMap, "context_block.mlp.fc1.weight");
             ModeLoaderlUtils.loadData(block.context_block.mlp.linear1.bias, weightMap, "context_block.mlp.fc1.bias");
             ModeLoaderlUtils.loadData(block.context_block.mlp.linear2.weight, weightMap, "context_block.mlp.fc2.weight");
             ModeLoaderlUtils.loadData(block.context_block.mlp.linear2.bias, weightMap, "context_block.mlp.fc2.bias");
        }
        ModeLoaderlUtils.loadData(block.context_block.adaLN_modulation.weight, weightMap, "context_block.adaLN_modulation.1.weight");
        ModeLoaderlUtils.loadData(block.context_block.adaLN_modulation.bias, weightMap, "context_block.adaLN_modulation.1.bias");
        
        block.x_block.norm1.gamma = ModeLoaderlUtils.loadData(block.x_block.norm1.gamma, weightMap, 1, "x_block.norm1.weight"); 
        ModeLoaderlUtils.loadData(block.x_block.qLinerLayer.weight, weightMap, "x_block.attn.ql.weight");
        ModeLoaderlUtils.loadData(block.x_block.kLinerLayer.weight, weightMap, "x_block.attn.kl.weight");
        ModeLoaderlUtils.loadData(block.x_block.vLinerLayer.weight, weightMap, "x_block.attn.vl.weight");
        ModeLoaderlUtils.loadData(block.x_block.oLinerLayer.weight, weightMap, "x_block.attn.proj.weight");
        ModeLoaderlUtils.loadData(block.x_block.oLinerLayer.bias, weightMap, "x_block.attn.proj.bias");
        block.x_block.norm2.gamma = ModeLoaderlUtils.loadData(block.x_block.norm2.gamma, weightMap, 1, "x_block.norm2.weight"); 
        ModeLoaderlUtils.loadData(block.x_block.mlp.linear1.weight, weightMap, "x_block.mlp.fc1.weight");
        ModeLoaderlUtils.loadData(block.x_block.mlp.linear1.bias, weightMap, "x_block.mlp.fc1.bias");
        ModeLoaderlUtils.loadData(block.x_block.mlp.linear2.weight, weightMap, "x_block.mlp.fc2.weight");
        ModeLoaderlUtils.loadData(block.x_block.mlp.linear2.bias, weightMap, "x_block.mlp.fc2.bias");
        ModeLoaderlUtils.loadData(block.x_block.adaLN_modulation.weight, weightMap, "x_block.adaLN_modulation.1.weight");
        ModeLoaderlUtils.loadData(block.x_block.adaLN_modulation.bias, weightMap, "x_block.adaLN_modulation.1.bias");

    }
    
    public static void main(String[] args) {
    	
    	int N = 2;
    	int T = 4;
    	int W = 8;
    	int hd = 4;
    	
    	int TT = 3;
    	
    	int mlp_ratio = 4;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
    	DiTJoinBlock jb = new DiTJoinBlock(W, W, mlp_ratio, hd, T, TT, false, false, true, true, nn);
    	
        String weight = "D:\\models\\JointBlock.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), jb, true);
    	
        String inputPath = "D:\\models\\img_x.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, T, 1, W, true);
        ModeLoaderlUtils.loadData(input, datas, "img_x", 3);
        
        String textPath = "D:\\models\\text_x.json";
        Map<String, Object> textDatas = LagJsonReader.readJsonFileSmallWeight(textPath);
        Tensor txt = new Tensor(N, TT, 1, W, true);
        ModeLoaderlUtils.loadData(txt, textDatas, "text_x", 3);
        
        String cPath = "D:\\models\\c_mod.json";
        Map<String, Object> cDatas = LagJsonReader.readJsonFileSmallWeight(cPath);
        Tensor c = new Tensor(N, 1, 1, W, true);
        ModeLoaderlUtils.loadData(c, cDatas, "c_mod", 2);
        
        input.view(N * T, 1, 1, W);
        txt.view(N * TT, 1, 1, W);
        
        jb.forward(input, txt, c);
        
        jb.getOutput().showDM("x_out");
        
//        jb.context_block.getOutput().showDM("cx");
        
        String dinputPath = "D:\\models\\dimg_x.json";
        Map<String, Object> d_datas = LagJsonReader.readJsonFileSmallWeight(dinputPath);
        Tensor dinput = new Tensor(N, T, 1, W, true);
        ModeLoaderlUtils.loadData(dinput, d_datas, "dimg_x", 3);
        
//        String dtextPath = "D:\\models\\dtext_x.json";
//        Map<String, Object> d_textDatas = LagJsonReader.readJsonFileSmallWeight(dtextPath);
        Tensor dtxt = new Tensor(N, TT, 1, W, true);
//        ClipModelUtils.loadData(dtxt, d_textDatas, "dtext_x", 3);
        
        Tensor dc = new Tensor(N, 1, 1, W, true);
        
        jb.back(dinput, dtxt, dc);
        
        jb.diff.showDM("dx");
        jb.context_block.diff.showDM("dtxt");
        dc.showDM("dc");
        
        jb.forward(input, txt, c);
        jb.getOutput().showDM("x_out");
        dc.clearGPU();
        jb.back(dinput, dtxt, dc);
        jb.diff.showDM("dx");
        jb.context_block.diff.showDM("dtxt");
        dc.showDM("dc");
        
    }
    
}

