package com.omega.engine.nn.layer.dit.mmjit;

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
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTAttentionLayer2
 *
 * @author Administrator
 */
public class MMJiTBlock extends Layer {
    
    private boolean qkNorm = false;
    
    private int batchSize = 1;

    private int time;
    public int imgTime;
    private int textTime;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;

    private boolean bias = false;
    private boolean normParams = true;
    
    public JiTJoinBlockHead x_block;
    public JiTJoinBlockHead context_block;
    
    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private RoPEKernel ropeKernel;

    private Tensor x_rq;
    private Tensor x_rk;
    private Tensor cond_rq;
    private Tensor cond_rk;
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

    private Tensor dattn;
    
    private int[] shape;
    private int[] t_shape;

    public MMJiTBlock(int embedDim, int headNum, int imgTime, int textTime, boolean bias, boolean qkNorm, boolean normParams, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.imgTime = imgTime;
        this.textTime = textTime;
        this.time = imgTime + textTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.qkNorm = qkNorm;
        this.normParams = normParams;
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
        
    	x_block = new JiTJoinBlockHead(embedDim, imgTime, bias, qkNorm, false, normParams, network);
    	
    	context_block = new JiTJoinBlockHead(embedDim, textTime, bias, qkNorm, false, normParams, network);
    	
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
        }
        if(ropeKernel == null) {
        	ropeKernel = new RoPEKernel(cuda());
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
            this.x_rq.viewOrg();
            this.x_rk.viewOrg();
            this.q.viewOrg();
            this.k.viewOrg();
            this.v.viewOrg();
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
        }
        if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
//        	System.err.println("in init-shape");
        	shape = new int[] {batchSize, time, headNum, dk};
        	t_shape = new int[] {batchSize, headNum, time, dk};
            // [batch_size，time，head_num，d_k]
            this.x_rq = CUDAMemoryManager.getCache("mmdit_block_x_rq", batchSize, imgTime, headNum, dk);
            this.x_rk = CUDAMemoryManager.getCache("mmdit_block_x_rk", batchSize, imgTime, headNum, dk);
            
            this.cond_rq = CUDAMemoryManager.getCache("mmdit_block_cond_rq", batchSize, textTime, headNum, dk);
            this.cond_rk = CUDAMemoryManager.getCache("mmdit_block_cond_rk", batchSize, textTime, headNum, dk);
            
            this.q = CUDAMemoryManager.getCache("mmdit_block_q", batchSize, time, 1, embedDim);
            this.k = CUDAMemoryManager.getCache("mmdit_block_k", batchSize, time, 1, embedDim);
            this.v = CUDAMemoryManager.getCache("mmdit_block_v", batchSize, time, 1, embedDim);
            
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
   
            this.x_attn = CUDAMemoryManager.getCache("mmdit_block_x_attn", batchSize * imgTime, 1, 1, embedDim);
            this.context_attn = CUDAMemoryManager.getCache("mmdit_block_context_attn", batchSize * textTime, 1, 1, embedDim);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
//            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
        	this.dattn = CUDAMemoryManager.getCache("mmdit_block_dattn", batchSize, headNum, time, time);
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
    
    public void output(Tensor context, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
    	x_block.pre_attention(input);
        /**
         * apply RoPE
         */
        ropeKernel.forward2d_t(cos2d, sin2d, x_block.q(), x_rq, imgTime, headNum, dk);
        ropeKernel.forward2d_t(cos2d, sin2d, x_block.k(), x_rk, imgTime, headNum, dk);
        
    	context_block.pre_attention(context);
        ropeKernel.forward2d_t(cos1d, sin1d, context_block.q(), cond_rq, textTime, headNum, dk);
        ropeKernel.forward2d_t(cos1d, sin1d, context_block.k(), cond_rk, textTime, headNum, dk);
    	
    	attentionKernel.concat_channel_forward(cond_rq, x_rq, q, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(cond_rk, x_rk, k, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(context_block.v(), x_block.v(), v, batchSize, textTime, imgTime, 1, embedDim);

        Tensor_OP().permute(q, qt, shape, t_shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(k, kt, shape, t_shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(v, vt, shape, t_shape, new int[]{0, 2, 1, 3});
        
        scaledDotProductAttention(qt, kt, vt);
        
        attentionKernel.unpermute(temp, oi, batchSize, time, headNum, dk);
        attentionKernel.concat_channel_backward(oi, context_attn, x_attn, batchSize, textTime, imgTime, 1, embedDim);
        
        x_block.post_attention(x_attn);
        context_block.post_attention(context_attn);

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
    
    public void diff(Tensor dcx, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
    	//recomplate the active status
    	attentionKernel.concat_channel_forward(context_block.q(), x_rq, q, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(context_block.k(), x_rk, k, batchSize, textTime, imgTime, 1, embedDim);
    	attentionKernel.concat_channel_forward(context_block.v(), x_block.v(), v, batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.concat_channel_backward(oi, context_attn, x_attn, batchSize, textTime, imgTime, 1, embedDim);
        ropeKernel.forward2d_t(cos2d, sin2d, x_block.q(), x_rq, imgTime, headNum, dk);
        ropeKernel.forward2d_t(cos2d, sin2d, x_block.k(), x_rk, imgTime, headNum, dk);
        ropeKernel.forward2d_t(cos1d, sin1d, context_block.q(), cond_rq, textTime, headNum, dk);
        ropeKernel.forward2d_t(cos1d, sin1d, context_block.k(), cond_rk, textTime, headNum, dk);
        
        Tensor x_diff = x_block.post_attention_back(delta);
        Tensor dattn = x_block.oLinerLayer.diff;

        Tensor cx_diff = context_block.post_attention_back(dcx);
        Tensor dattn_cx = context_block.oLinerLayer.diff;

        attentionKernel.concat_channel_forward(dattn_cx, dattn, oi, batchSize, textTime, imgTime, 1, embedDim);

        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);

        scaledDotProductAttentionBackward();
        
        Tensor_OP().permute(q, qt, t_shape, shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(k, kt, t_shape, shape, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(v, vt, t_shape, shape, new int[]{0, 2, 1, 3});
        
        attentionKernel.concat_channel_backward(qt, context_block.q(), x_block.q(), batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.concat_channel_backward(kt, context_block.k(), x_block.k(), batchSize, textTime, imgTime, 1, embedDim);
        attentionKernel.concat_channel_backward(vt, context_block.v(), x_block.v(), batchSize, textTime, imgTime, 1, embedDim);
        
        /**
         * RoPE backward
         */
        ropeKernel.backward2d_t(cos1d, sin1d, context_block.q(), cond_rq, textTime, headNum, dk);
        ropeKernel.backward2d_t(cos1d, sin1d, context_block.k(), cond_rk, textTime, headNum, dk);
        context_block.pre_attention_back(cx_diff, cond_rq, cond_rk);
        
        /**
         * RoPE backward
         */
        ropeKernel.backward2d_t(cos2d, sin2d, x_block.q(), x_rq, imgTime, headNum, dk);
        ropeKernel.backward2d_t(cos2d, sin2d, x_block.k(), x_rk, imgTime, headNum, dk);
        x_block.pre_attention_back(x_diff, x_rq, x_rk);
        
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
    
    public void forward(Tensor input, Tensor context, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
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
        this.output(context, cos1d, sin1d, cos2d, sin2d);
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
    
    public void back(Tensor delta, Tensor cx_delta, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cx_delta, cos1d, sin1d, cos2d, sin2d);
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
    
    public static void loadWeight(Map<String, Object> weightMap, MMJiTBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
       
    }
    
    public static void main(String[] args) {
    	
//    	int N = 2;
//    	int T = 4;
//    	int W = 8;
//    	int hd = 4;
//    	
//    	int TT = 3;
//    	
//    	float mlp_ratio = 4.0f;
//    	
//    	CNN nn = new CNN(null);
//        nn.CUDNN = true;
//        nn.number = N;
//    	
//    	DiTJoinBlockRoPE jb = new DiTJoinBlockRoPE(W, W, mlp_ratio, hd, T, TT, false, false, true, nn);
//    	
//        String weight = "D:\\models\\JointBlock.json";
//        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), jb, true);
//    	
//        String inputPath = "D:\\models\\img_x.json";
//        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//        Tensor input = new Tensor(N, T, 1, W, true);
//        ClipModelUtils.loadData(input, datas, "img_x", 3);
//        
//        String textPath = "D:\\models\\text_x.json";
//        Map<String, Object> textDatas = LagJsonReader.readJsonFileSmallWeight(textPath);
//        Tensor txt = new Tensor(N, TT, 1, W, true);
//        ClipModelUtils.loadData(txt, textDatas, "text_x", 3);
//        
//        String cPath = "D:\\models\\c_mod.json";
//        Map<String, Object> cDatas = LagJsonReader.readJsonFileSmallWeight(cPath);
//        Tensor c = new Tensor(N, 1, 1, W, true);
//        ClipModelUtils.loadData(c, cDatas, "c_mod", 2);
//        
//        input.view(N * T, 1, 1, W);
//        txt.view(N * TT, 1, 1, W);
//        
//        jb.forward(input, txt, c);
//        
//        jb.getOutput().showDM("x_out");
//        
////        jb.context_block.getOutput().showDM("cx");
//        
//        String dinputPath = "D:\\models\\dimg_x.json";
//        Map<String, Object> d_datas = LagJsonReader.readJsonFileSmallWeight(dinputPath);
//        Tensor dinput = new Tensor(N, T, 1, W, true);
//        ClipModelUtils.loadData(dinput, d_datas, "dimg_x", 3);
//        
////        String dtextPath = "D:\\models\\dtext_x.json";
////        Map<String, Object> d_textDatas = LagJsonReader.readJsonFileSmallWeight(dtextPath);
//        Tensor dtxt = new Tensor(N, TT, 1, W, true);
////        ClipModelUtils.loadData(dtxt, d_textDatas, "dtext_x", 3);
//        
//        Tensor dc = new Tensor(N, 1, 1, W, true);
//        
//        jb.back(dinput, dtxt, dc);
//        
//        jb.diff.showDM("dx");
//        jb.context_block.diff.showDM("dtxt");
//        dc.showDM("dc");
//        
//        jb.forward(input, txt, c);
//        jb.getOutput().showDM("x_out");
//        dc.clearGPU();
//        jb.back(dinput, dtxt, dc);
//        jb.diff.showDM("dx");
//        jb.context_block.diff.showDM("dtxt");
//        dc.showDM("dc");
        
    }
    
}

