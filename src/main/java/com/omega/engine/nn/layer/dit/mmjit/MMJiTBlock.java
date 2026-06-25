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

    private Tensor x_qt;
    private Tensor x_kt;
    private Tensor x_vt;
    private Tensor cond_qt;
    private Tensor cond_kt;
    private Tensor cond_vt;
    private Tensor x_rq;
    private Tensor x_rk;
    private Tensor cond_rq;
    private Tensor cond_rk;
    private Tensor q;
    private Tensor k;
    private Tensor v;
   
    private Tensor temp;
    private Tensor attn;
    private Tensor oi;
    
    private Tensor x_attn;
    private Tensor context_attn;

    private Tensor dattn;
    
    private Tensor dquery;
    private Tensor dkey;
    private Tensor dvalue;
    
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
        if (this.q != null) {
            this.x_rq.viewOrg();
            this.x_rk.viewOrg();
            this.q.viewOrg();
            this.k.viewOrg();
            this.v.viewOrg();
        }
        if (this.q == null || this.q.number != this.batchSize || this.q.height != this.time) {
            // [batch_size，time，head_num，d_k]
            this.x_rq = CUDAMemoryManager.getCache("mmdit_block_x_rq", batchSize, imgTime, headNum, dk);
            this.x_rk = CUDAMemoryManager.getCache("mmdit_block_x_rk", batchSize, imgTime, headNum, dk);
            
            this.cond_rq = CUDAMemoryManager.getCache("mmdit_block_cond_rq", batchSize, textTime, headNum, dk);
            this.cond_rk = CUDAMemoryManager.getCache("mmdit_block_cond_rk", batchSize, textTime, headNum, dk);
            
            this.q = Tensor.createGPUTensor(this.q, batchSize, time, 1, embedDim, true);
            this.k = Tensor.createGPUTensor(this.k, batchSize, time, 1, embedDim, true);
            this.v = Tensor.createGPUTensor(this.v, batchSize, time, 1, embedDim, true);
            
            this.x_qt = CUDAMemoryManager.getCache("mmdit_block_x_qt", batchSize, headNum, imgTime, dk);
            this.x_kt = CUDAMemoryManager.getCache("mmdit_block_x_kt", batchSize, headNum, imgTime, dk);
            this.x_vt = CUDAMemoryManager.getCache("mmdit_block_x_vt", batchSize, headNum, imgTime, dk);
            this.cond_qt = CUDAMemoryManager.getCache("mmdit_block_cond_qt", batchSize, headNum, textTime, dk);
            this.cond_kt = CUDAMemoryManager.getCache("mmdit_block_cond_kt", batchSize, headNum, textTime, dk);
            this.cond_vt = CUDAMemoryManager.getCache("mmdit_block_cond_vt", batchSize, headNum, textTime, dk);

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
            this.dquery = CUDAMemoryManager.getCache("mmdit_block_dquery", batchSize, headNum, time, dk);
            this.dkey = CUDAMemoryManager.getCache("mmdit_block_dkey", batchSize, headNum, time, dk);
            this.dvalue = CUDAMemoryManager.getCache("mmdit_block_dvalue", batchSize, headNum, time, dk);
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
    	int[] x_shape = new int[] {batchSize, imgTime, headNum, dk};
    	int[] x_t_shape = new int[] {batchSize, headNum, imgTime, dk};
    	Tensor_OP().permute(x_block.q(), x_qt, x_shape, x_t_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(x_block.k(), x_kt, x_shape, x_t_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(x_block.v(), x_vt, x_shape, x_t_shape, new int[]{0, 2, 1, 3});
        ropeKernel.rope_2d_rotate_half(cos2d, sin2d, x_qt, x_rq, imgTime, headNum, dk);
        ropeKernel.rope_2d_rotate_half(cos2d, sin2d, x_kt, x_rk, imgTime, headNum, dk);

    	context_block.pre_attention(context);
    	int[] cond_shape = new int[] {batchSize, textTime, headNum, dk};
    	int[] cond_t_shape = new int[] {batchSize, headNum, textTime, dk};
    	Tensor_OP().permute(context_block.q(), cond_qt, cond_shape, cond_t_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(context_block.k(), cond_kt, cond_shape, cond_t_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(context_block.v(), cond_vt, cond_shape, cond_t_shape, new int[]{0, 2, 1, 3});
    	ropeKernel.rope_2d_rotate_half(cos1d, sin1d, cond_qt, cond_rq, textTime, headNum, dk);
        ropeKernel.rope_2d_rotate_half(cos1d, sin1d, cond_kt, cond_rk, textTime, headNum, dk);
        
    	attentionKernel.cat_4d_dynamic_dim(cond_rq, x_rq, q, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
    	attentionKernel.cat_4d_dynamic_dim(cond_rk, x_rk, k, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
    	attentionKernel.cat_4d_dynamic_dim(cond_vt, x_vt, v, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
        
        scaledDotProductAttention(q, k, v);
        
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
    
    public void scaledDotProductAttentionBackward(Tensor query, Tensor key, Tensor value) {
        Tensor dvaccum = temp;
        /**
         * backward into dattn[b, nh, t, t2]
         * vt[b, nh, t2, dk] -> [b, nh, dk, t2]
         * dvaccum[b, nh, t, dk]
         */
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, value.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
        /**
         * backward into dvt[b, nh, t2, dk]
         * dvaccum[b, nh, t, dk]
         * attn[b, nh, t, t2] -> [b, nh, t2, t]
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dvalue.getGpuData(), dk, time * dk, batchSize * headNum);

        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);
        //		dattn.showDM();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        /**
         * backward into dqt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, key.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dquery.getGpuData(), dk, time * dk, batchSize * headNum);
        /**
         * backward into dkt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, query.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkey.getGpuData(), dk, time * dk, batchSize * headNum);
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
    	/**
    	 * recompate x_attn context_attn
    	 * x_attn
    	 * context_attn
    	 * is x txt oLinear input
    	 */
        attentionKernel.concat_channel_backward(oi, context_attn, x_attn, batchSize, textTime, imgTime, 1, embedDim);
        
        Tensor x_diff = x_block.post_attention_back(delta);
        Tensor dattn = x_block.oLinerLayer.diff;

        Tensor cx_diff = context_block.post_attention_back(dcx);
        Tensor dattn_cx = context_block.oLinerLayer.diff;

        attentionKernel.concat_channel_forward(dattn_cx, dattn, oi, batchSize, textTime, imgTime, 1, embedDim);

        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);

        scaledDotProductAttentionBackward(q, k, v);
        
    	attentionKernel.cat_4d_dynamic_dim_backward(dquery, cond_rq, x_rq, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
    	attentionKernel.cat_4d_dynamic_dim_backward(dkey, cond_rk, x_rk, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
    	attentionKernel.cat_4d_dynamic_dim_backward(dvalue, cond_vt, x_vt, batchSize, headNum, textTime, dk, batchSize, headNum, imgTime, dk, 2);
        
        /**
         * RoPE backward
         */
    	ropeKernel.rope_2d_back_rotate_half(cos1d, sin1d, cond_rq, cond_qt, textTime, headNum, dk);
        ropeKernel.rope_2d_back_rotate_half(cos1d, sin1d,cond_rk,  cond_kt, textTime, headNum, dk);
        int[] cond_shape = new int[] {batchSize, textTime, headNum, dk};
    	int[] cond_t_shape = new int[] {batchSize, headNum, textTime, dk};
    	Tensor_OP().permute(cond_qt, context_block.q(), cond_t_shape, cond_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(cond_kt, context_block.k(), cond_t_shape, cond_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(cond_vt, context_block.v(), cond_t_shape, cond_shape, new int[]{0, 2, 1, 3});
        context_block.pre_attention_back(cx_diff, context_block.q(), context_block.k(), context_block.v());
        
        /**
         * RoPE backward
         */
    	ropeKernel.rope_2d_back_rotate_half(cos2d, sin2d, x_rq, x_qt, imgTime, headNum, dk);
        ropeKernel.rope_2d_back_rotate_half(cos2d, sin2d, x_rk,  x_kt, imgTime, headNum, dk);
    	int[] x_shape = new int[] {batchSize, imgTime, headNum, dk};
    	int[] x_t_shape = new int[] {batchSize, headNum, imgTime, dk};
    	Tensor_OP().permute(x_qt, x_block.q(), x_t_shape, x_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(x_kt, x_block.k(), x_t_shape, x_shape, new int[]{0, 2, 1, 3});
    	Tensor_OP().permute(x_vt, x_block.v(), x_t_shape, x_shape, new int[]{0, 2, 1, 3});
    	x_block.pre_attention_back(x_diff, x_block.q(), x_block.k(), x_block.v());
        
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
        
        block.x_block.norm1.gamma = ModeLoaderlUtils.loadData(block.x_block.norm1.gamma, weightMap, 1, "img_norm1.weight"); 
        block.x_block.norm2.gamma = ModeLoaderlUtils.loadData(block.x_block.norm2.gamma, weightMap, 1, "img_norm2.weight"); 
        block.context_block.norm1.gamma = ModeLoaderlUtils.loadData(block.context_block.norm1.gamma, weightMap, 1, "txt_norm1.weight"); 
        block.context_block.norm2.gamma = ModeLoaderlUtils.loadData(block.context_block.norm2.gamma, weightMap, 1, "txt_norm2.weight"); 
        
        ModeLoaderlUtils.loadData(block.x_block.qLinerLayer.weight, weightMap, "img_q.weight");
        ModeLoaderlUtils.loadData(block.x_block.qLinerLayer.bias, weightMap, "img_q.bias");
        ModeLoaderlUtils.loadData(block.x_block.kLinerLayer.weight, weightMap, "img_k.weight");
        ModeLoaderlUtils.loadData(block.x_block.kLinerLayer.bias, weightMap, "img_k.bias");
        ModeLoaderlUtils.loadData(block.x_block.vLinerLayer.weight, weightMap, "img_v.weight");
        ModeLoaderlUtils.loadData(block.x_block.vLinerLayer.bias, weightMap, "img_v.bias");
        
        ModeLoaderlUtils.loadData(block.context_block.qLinerLayer.weight, weightMap, "txt_q.weight");
        ModeLoaderlUtils.loadData(block.context_block.qLinerLayer.bias, weightMap, "txt_q.bias");
        ModeLoaderlUtils.loadData(block.context_block.kLinerLayer.weight, weightMap, "txt_k.weight");
        ModeLoaderlUtils.loadData(block.context_block.kLinerLayer.bias, weightMap, "txt_k.bias");
        ModeLoaderlUtils.loadData(block.context_block.vLinerLayer.weight, weightMap, "txt_v.weight");
        ModeLoaderlUtils.loadData(block.context_block.vLinerLayer.bias, weightMap, "txt_v.bias");
        
        ModeLoaderlUtils.loadData(block.x_block.oLinerLayer.weight, weightMap, "img_proj.weight");
        ModeLoaderlUtils.loadData(block.x_block.oLinerLayer.bias, weightMap, "img_proj.bias");
        
        ModeLoaderlUtils.loadData(block.context_block.oLinerLayer.weight, weightMap, "txt_proj.weight");
        ModeLoaderlUtils.loadData(block.context_block.oLinerLayer.bias, weightMap, "txt_proj.bias");
        
        ModeLoaderlUtils.loadData(block.x_block.mlp.w12.weight, weightMap, "img_mlp.w12.weight");
        ModeLoaderlUtils.loadData(block.x_block.mlp.w3.weight, weightMap, "img_mlp.w3.weight");
        
        ModeLoaderlUtils.loadData(block.context_block.mlp.w12.weight, weightMap, "txt_mlp.w12.weight");
        ModeLoaderlUtils.loadData(block.context_block.mlp.w3.weight, weightMap, "txt_mlp.w3.weight");
    }
    
    public static void main(String[] args) {
    	
    	int B = 2;
    	int N = 256;
    	int DM = 768;
    	int hn = 12;
    	
    	int TT = 77;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
        MMJiTBlock jb = new MMJiTBlock(DM, hn, N, TT, true, false, true, nn);
    	
        String weight = "D:\\models\\mmjit.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), jb, true);
    	
	    String inputPath = "D:\\models\\mmjit_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(B, N, 1, DM, true);
        ModeLoaderlUtils.loadData(input, datas, "x", 3);
    	
	    String cyPath = "D:\\models\\mmjit_txt.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor txt = new Tensor(B, TT, 1, DM, true);
	    ModeLoaderlUtils.loadData(txt, cydatas, "txt", 3);

        input.view(B * N, 1, 1, DM);
        txt.view(B * TT, 1, 1, DM);
        
        int dk = DM / hn;
        
        int grid = (int) Math.sqrt(N);
        
        int theta = 10000;
        
        Tensor[] cs1d = RoPEKernel.create1DRope(TT, dk, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];
//        cos1d.showDMByOffset(50 * 64, 64, "cos1d");
//        sin1d.showDMByOffset(50 * 64, 64, "sin1d");
        Tensor[] cs2d = RoPEKernel.create2DRope(hn, N, dk, grid, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];
//        cos2d.showDMByOffset(100 * 64, 64, "cos2d");
//        sin2d.showDMByOffset(100 * 64, 64, "sin2d");
        
	    String dinputPath = "D:\\models\\mmjit_delta.json";
	    Map<String, Object> d_datas = LagJsonReader.readJsonFileSmallWeight(dinputPath);
	    Tensor delta = new Tensor(B, TT+N, 1, DM, true);
	    ModeLoaderlUtils.loadData(delta, d_datas, "delta", 3);
        
	    Tensor img_delta = new Tensor(B, N, 1, DM, true);
	    Tensor txt_delta = new Tensor(B, TT, 1, DM, true);
	    
	    jb.Tensor_OP().op.concat_channel_backward(delta, txt_delta, img_delta, B, TT, N, 1, DM);
	    
	    for(int i = 0;i<10;i++){

	        jb.forward(input, txt, cos1d, sin1d, cos2d, sin2d);
	        
	        jb.getOutput().showDM("x_out");
	        jb.context_block.getOutput().showDM("txt_out");
	        
//		    txt_delta.showDM("txt_delta");
//		    img_delta.showDM("img_delta");
		    
		    jb.back(img_delta, txt_delta, cos1d, sin1d, cos2d, sin2d);
		    
		    jb.diff.showDM("dx");
		    jb.context_block.diff.showDM("txt_dx");
		    
	    }
        
    }
    
}

