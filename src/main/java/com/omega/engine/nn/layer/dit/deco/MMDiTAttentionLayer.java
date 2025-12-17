package com.omega.engine.nn.layer.dit.deco;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * MMDiTAttentionLayer
 *
 * @author Administrator
 */
public class MMDiTAttentionLayer extends Layer {
	
    public RMSLayer qNorm;
    public RMSLayer kNorm;
    public RMSLayer kyNorm;
	
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    
    public FullyLayer kyLinerLayer;
    public FullyLayer vyLinerLayer;
    
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    
    private int txtTime;
    
    //	public GNLayer gn;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private RoPEKernel ropeKernel;
    
    private Tensor y_tmp;
    
    private Tensor k_tmp;
    private Tensor v_tmp;
    
    private Tensor rq;
    private Tensor rk;
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
    
    private boolean qkNorm = false;
    
//    private Tensor temp_out;
    
    private int[] p_0213 = new int[]{0, 2, 1, 3};
    
    private Tensor diff_ky;
    private Tensor diff_vy;

    public MMDiTAttentionLayer(int embedDim, int headNum, int time, int txtTime, boolean bias, boolean qkNorm) {
        this.bias = bias;
        this.time = time;
        this.txtTime = txtTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.qkNorm = qkNorm;
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

    public MMDiTAttentionLayer(int embedDim, int headNum, int time, int txtTime, boolean bias, boolean qkNorm, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.txtTime = txtTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        if (embedDim % headNum != 0) {
            throw new RuntimeException("embedDim % headNum must be zero.");
        }
        this.qkNorm = qkNorm;
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
       
    	if(qkNorm) {
        	qNorm = new RMSLayer(1, 1, dk, true, BNType.fully_bn, network);
        	kNorm = new RMSLayer(1, 1, dk, true, BNType.fully_bn, network);
        	kyNorm = new RMSLayer(1, 1, dk, true, BNType.fully_bn, network);
        }
    	
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.qLinerLayer.weight, 1, embedDim, embedDim);
        if(this.qLinerLayer.bias != null) {
        	this.qLinerLayer.bias.clearGPU();
        }

        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.kLinerLayer.weight, 1, embedDim, embedDim);
        if(this.kLinerLayer.bias != null) {
        	this.kLinerLayer.bias.clearGPU();
        }

        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.vLinerLayer.weight, 1, embedDim, embedDim);
        if(this.vLinerLayer.bias != null) {
        	this.vLinerLayer.bias.clearGPU();
        }

        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.oLinerLayer.weight, 1, embedDim, embedDim);
        if(this.oLinerLayer.bias != null) {
        	this.oLinerLayer.bias.clearGPU();
        }

        this.kyLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        RandomUtils.xavier_uniform(this.kyLinerLayer.weight, 1, embedDim, embedDim);
        if(this.kyLinerLayer.bias != null) {
        	this.kyLinerLayer.bias.clearGPU();
        }
        this.vyLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
        RandomUtils.xavier_uniform(this.vyLinerLayer.weight, 1, embedDim, embedDim);
        if(this.vyLinerLayer.bias != null) {
        	this.vyLinerLayer.bias.clearGPU();
        }
        
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel((time + txtTime), 1, 1, cuda());
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
        this.number = input.number;
        this.batchSize = this.number/time;
        if (this.qt != null) {
            this.output.viewOrg();
            this.qt.viewOrg();
            this.kt.viewOrg();
            this.vt.viewOrg();
            this.oi.viewOrg();
            this.rq.viewOrg();
            this.rk.viewOrg();
            temp.clearGPU();
        }
        if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
            // [batch_size，time，head_num，d_k]
        	this.rq = Tensor.createGPUTensor(this.rq, batchSize, headNum, time, dk, true);
            this.rk = Tensor.createGPUTensor(this.rk, batchSize, headNum, time, dk, true);
            this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
            this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
            this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
            
            this.y_tmp = CUDAMemoryManager.getCache("dit_block_attn_y_tmp", batchSize, headNum, txtTime, dk);
            
            this.k_tmp = CUDAMemoryManager.getCache("dit_block_attn_k_tmp", batchSize, headNum, txtTime + time, dk);
            this.v_tmp = CUDAMemoryManager.getCache("dit_block_attn_v_tmp", batchSize, headNum, txtTime + time, dk);
            
            // [batch_size，n_heads，len_q，len_k]
            if ((time + txtTime) < dk) {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
            } else {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, (time + txtTime), true);
            }
            // [batch_size，n_heads，len_q，len_k]
            this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, (time + txtTime), true);
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
//            this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
        }
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
            this.getoLinerLayer().getOutput().viewOrg();
            qt.viewOrg();
            kt.viewOrg();
            vt.viewOrg();
        }
        if(qkNorm && this.qNorm.getOutput() != null) {
        	this.qNorm.getOutput().viewOrg();
        	this.kNorm.getOutput().viewOrg();
        	this.kyNorm.getOutput().viewOrg();
        }
    }
    
    public void init_eval(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number/time;
        this.rq = CUDAMemoryManager.getCache("dit_block_attn_rq", batchSize, headNum, time, dk);
    	this.rk = CUDAMemoryManager.getCache("dit_block_attn_rk", batchSize, headNum, time, dk);
    	this.qt = CUDAMemoryManager.getCache("dit_block_attn_qt", batchSize, headNum, time, dk);
    	this.kt = CUDAMemoryManager.getCache("dit_block_attn_kt", batchSize, headNum, time, dk);
    	this.vt = CUDAMemoryManager.getCache("dit_block_attn_vt", batchSize, headNum, time, dk);
    	
    	this.y_tmp = CUDAMemoryManager.getCache("dit_block_attn_y_tmp", batchSize, headNum, txtTime, dk);
        this.k_tmp = CUDAMemoryManager.getCache("dit_block_attn_k_tmp", batchSize, headNum, txtTime + time, dk);
        this.v_tmp = CUDAMemoryManager.getCache("dit_block_attn_v_tmp", batchSize, headNum, txtTime + time, dk);
        // [batch_size，n_heads，len_q，len_k]
        if ((time + txtTime) < dk) {
            this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
        } else {
            this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, (time + txtTime), true);
        }
        temp.clearGPU();
        // [batch_size，n_heads，len_q，len_k]
        this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, (time + txtTime), true);
        // [batch_size, len_q, n_heads * dim_v]
        this.oi = CUDAMemoryManager.getCache("dit_block_attn_oi", batchSize * time, 1, 1, embedDim);
//        this.output = CUDAMemoryManager.getCache("dit_block_attn_out", input.number, input.channel, input.height, input.width);
//        this.temp_out = CUDAMemoryManager.getCache("dit_block_attn_temp_out", input.number, input.channel, input.height, input.width);
        if(qkNorm && this.qNorm.getOutput() != null) {
        	this.qNorm.getOutput().viewOrg();
        	this.kNorm.getOutput().viewOrg();
        	this.kyNorm.getOutput().viewOrg();
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
        	this.dqt = CUDAMemoryManager.getCache("cache_dqt", batchSize, headNum, time, dk);
        	this.dkt = CUDAMemoryManager.getCache("cache_dkt", batchSize, headNum, (time + txtTime), dk);
        	this.dvt = CUDAMemoryManager.getCache("cache_dvt", batchSize, headNum, (time + txtTime), dk);
        	this.dattn = CUDAMemoryManager.getCache("cache_dattn", batchSize, headNum, time, (time + txtTime));
        	this.diff_ky = CUDAMemoryManager.getCache("diff_ky", batchSize * txtTime, 1, 1, embedDim);
        	this.diff_vy = CUDAMemoryManager.getCache("diff_vy", batchSize * txtTime, 1, 1, embedDim);
        } else {
        	this.dqt.viewOrg(batchSize, headNum, time, dk);
        	this.dkt.viewOrg(batchSize, headNum, (time + txtTime), dk);
        	this.dvt.viewOrg(batchSize, headNum, (time + txtTime), dk);
        	this.dattn.viewOrg(batchSize, headNum, time, (time + txtTime));
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
    
    public void output(Tensor context, Tensor pos) {
        // TODO Auto-generated method stub
//    	input.showDM("input");
//    	context.showDM("context");
    	
        this.getqLinerLayer().forward(input);
        this.getkLinerLayer().forward(input);
        this.getvLinerLayer().forward(input);
        
        this.kyLinerLayer.forward(context);
        this.vyLinerLayer.forward(context);
        
        Tensor qx = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor kx = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor vx = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);

        Tensor ky = kyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
        Tensor vy = vyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
        
        Tensor_OP().permute(qx, qt, p_0213);
        Tensor_OP().permute(kx, kt, p_0213);
        Tensor_OP().permute(vx, vt, p_0213);

        Tensor q = qt;
        Tensor k = kt;
        if(qkNorm) {
        	qNorm.forward(q);
        	kNorm.forward(k);
        	q = qNorm.getOutput();
        	k = kNorm.getOutput();
        }

        /**
         * apply RoPE
         * qt = [B, HN, T, HS]
         */
//        q.showDM("q");
        ropeKernel.apply_rotary_emb(q, pos, rq);
        ropeKernel.apply_rotary_emb(k, pos, rk);
        
        Tensor_OP().permute(ky, y_tmp, p_0213);
        Tensor kyt = y_tmp;
        if(qkNorm) {
        	kyNorm.forward(kyt);
        	kyt = kyNorm.getOutput();
        }
        attentionKernel.concat_height_forward(rk, kyt, k_tmp, batchSize, headNum, time, txtTime, dk);
        
        Tensor_OP().permute(vy, y_tmp, p_0213);
        Tensor vyt = y_tmp;
        attentionKernel.concat_height_forward(vt, vyt, v_tmp, batchSize, headNum, time, txtTime, dk);
        
//        rq.showDM("rq");
        scaledDotProductAttention(rq, k_tmp, v_tmp);
        
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output_eval(Tensor context, Tensor pos) {
        // TODO Auto-generated method stub
    	this.getqLinerLayer().forward(input);
        this.getkLinerLayer().forward(input);
        this.getvLinerLayer().forward(input);
        
        this.kyLinerLayer.forward(context);
        this.vyLinerLayer.forward(context);
        
        Tensor qx = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor kx = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor vx = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);

        Tensor ky = kyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
        Tensor vy = vyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
        
        Tensor_OP().permute(qx, qt, p_0213);
        Tensor_OP().permute(kx, kt, p_0213);
        Tensor_OP().permute(vx, vt, p_0213);

        Tensor q = qt;
        Tensor k = kt;
        if(qkNorm) {
        	qNorm.forward(q);
        	kNorm.forward(k);
        	q = qNorm.getOutput();
        	k = kNorm.getOutput();
        }

        /**
         * apply RoPE
         * qt = [B, HN, T, HS]
         */
//        q.showDM("q");
        ropeKernel.apply_rotary_emb(q, pos, rq);
        ropeKernel.apply_rotary_emb(k, pos, rk);
        
        Tensor_OP().permute(ky, y_tmp, p_0213);
        Tensor kyt = y_tmp;
        if(qkNorm) {
        	kyNorm.forward(kyt);
        	kyt = kyNorm.getOutput();
        }
        attentionKernel.concat_height_forward(rk, kyt, k_tmp, batchSize, headNum, time, txtTime, dk);
        
        Tensor_OP().permute(vy, y_tmp, p_0213);
        Tensor vyt = y_tmp;
        attentionKernel.concat_height_forward(vt, vyt, v_tmp, batchSize, headNum, time, txtTime, dk);
        
//        rq.showDM("rq");
        scaledDotProductAttention(rq, k_tmp, v_tmp);
        
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }

    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;

        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, (time + txtTime), time, dk, 1.0f, key.getGpuData(), dk, (time + txtTime) * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), (time + txtTime), time * (time + txtTime), batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);

        softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        Tensor tmp = attn;

        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, (time + txtTime), 1.0f, value.getGpuData(), dk, (time + txtTime) * dk, tmp.getGpuData(), (time + txtTime), time * (time + txtTime), 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
//        vaccum.showDMByOffsetRed(0, batchSize * 12 * 256 * 64, "vaccum");
    }

    public void scaledDotProductAttentionBackward(Tensor q, Tensor k, Tensor value) {
        Tensor tmp = attn;

        Tensor dvaccum = temp;
        /**
         * backward into dattn[b, nh, t, t2]
         * vt[b, nh, t2, dk] -> [b, nh, dk, t2]
         * dvaccum[b, nh, t, dk]
         */
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, (time + txtTime), time, dk, 1.0f, value.getGpuData(), dk, (time + txtTime) * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), (time + txtTime), time * (time + txtTime), batchSize * headNum);

        /**
         * backward into dvt[b, nh, t2, dk]
         * dvaccum[b, nh, t, dk]
         * attn[b, nh, t, t2] -> [b, nh, t2, t]
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, (time + txtTime), time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), (time + txtTime), time * (time + txtTime), 0.0f, dvt.getGpuData(), dk, (time + txtTime) * dk, batchSize * headNum);

        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);

//        dattn.showDM("dattn");

        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        
        /**
         * backward into dqt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, (time + txtTime), 1.0f, k.getGpuData(), dk, (time + txtTime) * dk, dpreatt.getGpuData(), (time + txtTime), time * (time + txtTime), 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);

        /**
         * backward into dkt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, (time + txtTime), time, 1.0f, q.getGpuData(), dk, time * dk, dpreatt.getGpuData(), (time + txtTime), time * (time + txtTime), 0.0f, dkt.getGpuData(), dk, (time + txtTime) * dk, batchSize * headNum);
        
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
    
    public void diff(Tensor dcontext, Tensor pos) {
    	 this.getoLinerLayer().back(delta, oi);
         attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
         scaledDotProductAttentionBackward(rq, k_tmp, v_tmp);

         attentionKernel.concat_height_backward(dvt, vt, y_tmp, batchSize, headNum, time, txtTime, dk);
         
         Tensor dvy = vyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
         Tensor_OP().permute(y_tmp, dvy, p_0213);
         
         attentionKernel.concat_height_backward(dkt, kt, y_tmp, batchSize, headNum, time, txtTime, dk);

         Tensor dkyt = y_tmp;
         
         if(qkNorm) {
        	 kyNorm.back(dkyt);
        	 dkyt = kyNorm.diff;
         }
         
         Tensor dky = kyLinerLayer.getOutput().view(batchSize, txtTime, headNum, dk);
         Tensor_OP().permute(dkyt, dky, p_0213);

         /**
          * RoPE backward
          */
         ropeKernel.apply_rotary_emb_back(dqt, pos, rq);
         ropeKernel.apply_rotary_emb_back(kt, pos, rk);

         Tensor dqt_d = rq;
         Tensor dkt_d = rk;
         
         Tensor_OP().permute(this.getkLinerLayer().getOutput(), kt, p_0213);
         
         if(qkNorm) {
          	qNorm.back(dqt_d);
          	kNorm.back(dkt_d);
          	dqt_d = qNorm.diff;
          	dkt_d = kNorm.diff;
         }

         qt.view(batchSize, time, headNum, dk);
         kt.view(batchSize, time, headNum, dk);
//         vt.view(batchSize, time, headNum, dk);
         
      	 Tensor_OP().permute(dqt_d, qt, p_0213);
         Tensor_OP().permute(dkt_d, kt, p_0213);

         Tensor queryDelta = qt.view(this.getqLinerLayer().getOutput().getOrgShape());
         this.getqLinerLayer().getOutput().viewOrg();
         this.getqLinerLayer().back(queryDelta, this.getqLinerLayer().getOutput());
         
         qt.view(batchSize, time, headNum, dk);
         Tensor_OP().permute(vt, qt, p_0213);
         
         Tensor keyDelta = kt.view(this.getkLinerLayer().getOutput().getOrgShape());
         Tensor valueDelta = qt.view(this.getvLinerLayer().getOutput().getOrgShape());
         
         this.getkLinerLayer().getOutput().viewOrg();
         this.getvLinerLayer().getOutput().viewOrg();
         this.getkLinerLayer().back(keyDelta, this.getkLinerLayer().getOutput());
         this.getvLinerLayer().back(valueDelta, this.getvLinerLayer().getOutput());
         
         Tensor_OP().add(this.getqLinerLayer().diff, this.getkLinerLayer().diff, this.getqLinerLayer().diff);
         Tensor_OP().add(this.getqLinerLayer().diff, this.getvLinerLayer().diff, this.getqLinerLayer().diff);
         // dxt
         this.diff = this.getqLinerLayer().diff;
         
         dky = dky.view(this.kyLinerLayer.getOutput().getOrgShape());
         dvy = dvy.view(this.vyLinerLayer.getOutput().getOrgShape());
         
         kyLinerLayer.back(dky, diff_ky);
         vyLinerLayer.back(dvy, diff_vy);
         
         Tensor_OP().add(kyLinerLayer.diff, dcontext, dcontext);
         Tensor_OP().add(vyLinerLayer.diff, dcontext, dcontext);
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
    
    public void forward(Tensor input, Tensor context, Tensor pos) {
        // TODO Auto-generated method stub
    	if(network.RUN_MODEL == RunModel.EVAL) {
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
//            this.output_eval(pos);
    	}else {
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
            this.output(context, pos);
    	}  
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
    
    public void back(Tensor delta, Tensor dcontext,Tensor pos) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dcontext, pos);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    	if(qkNorm) {
	        qNorm.update();
	        kNorm.update();
	        kyNorm.update();
    	}
        getqLinerLayer().update();
        getkLinerLayer().update();
        getvLinerLayer().update();
        getoLinerLayer().update();
        
        kyLinerLayer.update();
        vyLinerLayer.update();
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
    	if(qkNorm) {
	        qNorm.saveModel(outputStream);
	        kNorm.saveModel(outputStream);
	        kyNorm.saveModel(outputStream);
    	}
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
        
        kyLinerLayer.saveModel(outputStream);
        vyLinerLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(qkNorm) {
	        qNorm.loadModel(inputStream, 1, 1, dk, BNType.fully_bn);
	        kNorm.loadModel(inputStream, 1, 1, dk, BNType.fully_bn);
	        kyNorm.loadModel(inputStream, 1, 1, dk, BNType.fully_bn);
    	}
        getqLinerLayer().loadModel(inputStream);
        getkLinerLayer().loadModel(inputStream);
        getvLinerLayer().loadModel(inputStream);
        getoLinerLayer().loadModel(inputStream);
        
        kyLinerLayer.loadModel(inputStream);
        vyLinerLayer.loadModel(inputStream);
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
    	if(qkNorm) {
	        qNorm.accGrad(scale);
	        kNorm.accGrad(scale);
	        kyNorm.accGrad(scale);
    	}
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
        
        kyLinerLayer.accGrad(scale);
        vyLinerLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, MMDiTAttentionLayer block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(block.qLinerLayer.weight, weightMap, "q_x.weight");
        ModeLoaderlUtils.loadData(block.qLinerLayer.bias, weightMap, "q_x.bias");
        ModeLoaderlUtils.loadData(block.kLinerLayer.weight, weightMap, "k_x.weight");
        ModeLoaderlUtils.loadData(block.kLinerLayer.bias, weightMap, "k_x.bias");
        ModeLoaderlUtils.loadData(block.vLinerLayer.weight, weightMap, "v_x.weight");
        ModeLoaderlUtils.loadData(block.vLinerLayer.bias, weightMap, "v_x.bias");
        ModeLoaderlUtils.loadData(block.oLinerLayer.weight, weightMap, "proj.weight");
        ModeLoaderlUtils.loadData(block.oLinerLayer.bias, weightMap, "proj.bias");
        
        ModeLoaderlUtils.loadData(block.kyLinerLayer.weight, weightMap, "k_y.weight");
        ModeLoaderlUtils.loadData(block.kyLinerLayer.bias, weightMap, "k_y.bias");
        ModeLoaderlUtils.loadData(block.vyLinerLayer.weight, weightMap, "v_y.weight");
        ModeLoaderlUtils.loadData(block.vyLinerLayer.bias, weightMap, "v_y.bias");
        
        block.qNorm.gamma = ModeLoaderlUtils.loadData(block.qNorm.gamma, weightMap, 1, "q_norm.weight"); 
        block.kNorm.beta = ModeLoaderlUtils.loadData(block.kNorm.beta, weightMap, 1, "k_norm.weight");
        block.kyNorm.gamma = ModeLoaderlUtils.loadData(block.kyNorm.gamma, weightMap, 1, "ky_norm.weight"); 

    }
    
    public static void main(String[] args) {
    	
    	int N = 2;
    	int T = 256;
    	int M = 768;
    	int HN = 12;
    	
    	int TT = 77;

    	Transformer tf = new Transformer();
        tf.number = N * T;
        tf.time = T;
    	
    	MMDiTAttentionLayer attn = new MMDiTAttentionLayer(M, HN, T, TT, true, true, tf);
    	
    	String weightPath = "D:\\models\\mmdit_weight.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(weightPath);
    	loadWeight(datas, attn, true);
    	
    	String inputPath = "D:\\models\\mmdit_x.json";
	    Map<String, Object> xdatas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, T, 1, M, true);
	    ModeLoaderlUtils.loadData(input, xdatas, "x", 3);
	    
    	String contextPath = "D:\\models\\mmdit_context.json";
	    Map<String, Object> cdatas = LagJsonReader.readJsonFileSmallWeight(contextPath);
	    Tensor context = new Tensor(N, TT, 1, M, true);
	    ModeLoaderlUtils.loadData(context, cdatas, "context", 3);
	    
	    Tensor pos = RoPEKernel.precompute_freqs_cis_2d_tensor(M / HN, 16, 16, 10000, 16);

        input.view(N * T, 1, 1, M);
        context.view(N * TT, 1, 1, M);
        
	    attn.forward(input, context, pos);
	    
	    attn.getOutput().showDM("output");
	    
    	String deltaPath = "D:\\models\\mmdit_delta.json";
	    Map<String, Object> ddatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
	    Tensor delta = new Tensor(N, T, 1, M, true);
	    ModeLoaderlUtils.loadData(delta, ddatas, "delta", 3);
	    delta.view(N * T, 1, 1, M);
	    
	    Tensor dcontext = new Tensor(N * TT, 1, 1, M, true); 
	    
	    attn.back(delta, dcontext, pos);
	    
	    attn.diff.showDM("diff");
	    
	    for(int i = 0;i<10;i++) {
		    attn.forward(input, context, pos);
		    attn.getOutput().showDM("output");
		    attn.back(delta, dcontext, pos);
		    attn.diff.showDM("diff");
	    }
	    
    }
    
}

