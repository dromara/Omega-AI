package com.omega.engine.nn.layer.dit.modules;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.gpu.CudnnFlashAttentionKernel;
import com.omega.engine.nn.layer.gpu.DiTQKVFusedKernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTAttentionLayer4
 *
 * @author Administrator
 */
public class DiTAttentionLayerQKVFused extends Layer {
	
    public RMSLayer qNorm;
    public RMSLayer kNorm;
	
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
    private RoPEKernel ropeKernel;
    private DiTQKVFusedKernel qkvFusedKernel;
    
    private CudnnFlashAttentionKernel cudnn_sdpa;
    
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
    private Tensor qkvDelta;
    private Tensor dattn;
    private int batchSize = 1;
    
    private boolean qkNorm = false;
    
    private Tensor temp_out;
    
    public DiTAttentionLayerQKVFused(int embedDim, int headNum, int time, boolean bias, boolean qkNorm) {
        this.bias = bias;
        this.time = time;
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

    public DiTAttentionLayerQKVFused(int embedDim, int headNum, int time, boolean bias, boolean qkNorm, Network network) {
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
        }
    	
        this.setqkvLinerLayer(new FullyLayer(embedDim, 3 * embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.qkvLinerLayer.weight, 1, embedDim, 3 * embedDim);
        if(this.qkvLinerLayer.bias != null) {
        	this.qkvLinerLayer.bias.clearGPU();
        }
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        RandomUtils.xavier_uniform(this.oLinerLayer.weight, 1, embedDim, embedDim);
//        this.oLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.oLinerLayer.bias != null) {
        	this.oLinerLayer.bias.clearGPU();
        }
        //		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(embedDim * embedDim, 0.1f, 0.01f), true);
        
        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        if (softmaxKernel == null) {
            softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
        }
        if (ropeKernel == null) {
            ropeKernel = new RoPEKernel(cuda());
        }
        if (qkvFusedKernel == null) {
            qkvFusedKernel = new DiTQKVFusedKernel(cuda());
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

        if(this.network.CUDNN_SDPA && cudnn_sdpa == null) {
        	cudnn_sdpa = new CudnnFlashAttentionKernel(batchSize, headNum, time, dk);
        }
        
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
            // [batch_size，n_heads，len_q，len_k]
            if (time < dk || this.network.CUDNN_SDPA) {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
            } else {
                this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, time, true);
            }
            if(!this.network.CUDNN_SDPA) {
            	 // [batch_size，n_heads，len_q，len_k]
                this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
            }
            // [batch_size, len_q, n_heads * dim_v]
            this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
            this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
        }
        if (this.getqkvLinerLayer().getOutput() != null) {
            this.getqkvLinerLayer().getOutput().viewOrg();
            this.getoLinerLayer().getOutput().viewOrg();
        }
        if(qkNorm && this.qNorm.getOutput() != null) {
        	this.qNorm.getOutput().viewOrg();
        	this.kNorm.getOutput().viewOrg();
        }
    }
    
    public void init_eval(Tensor input) {
        // TODO Auto-generated method stub
    	
        this.number = input.number;
        this.batchSize = this.number/time;

   	 	if(this.network.CUDNN_SDPA && cudnn_sdpa == null) {
        	cudnn_sdpa = new CudnnFlashAttentionKernel(batchSize, headNum, time, dk);
        }
   	
        this.rq = CUDAMemoryManager.getCache("dit_block_attn_rq", batchSize, headNum, time, dk);
    	this.rk = CUDAMemoryManager.getCache("dit_block_attn_rk", batchSize, headNum, time, dk);
    	this.qt = CUDAMemoryManager.getCache("dit_block_attn_qt", batchSize, headNum, time, dk);
    	this.kt = CUDAMemoryManager.getCache("dit_block_attn_kt", batchSize, headNum, time, dk);
    	this.vt = CUDAMemoryManager.getCache("dit_block_attn_vt", batchSize, headNum, time, dk);
        // [batch_size，n_heads，len_q，len_k]
		if (time < dk || this.network.CUDNN_SDPA) {
			this.temp = CUDAMemoryManager.getCache("dit_block_attn_temp", batchSize, headNum, time, dk);
		} else {
			this.temp = CUDAMemoryManager.getCache("dit_block_attn_temp", batchSize, headNum, time, time);
		}
		temp.clearGPU();
		if(!this.network.CUDNN_SDPA) {
		 	// [batch_size，n_heads，len_q，len_k]
		    this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
		}
        // [batch_size, len_q, n_heads * dim_v]
        this.oi = CUDAMemoryManager.getCache("dit_block_attn_oi", batchSize * time, 1, 1, embedDim);
        this.output = CUDAMemoryManager.getCache("dit_block_attn_out", input.number, input.channel, input.height, input.width);
        this.temp_out = CUDAMemoryManager.getCache("dit_block_attn_qkv_temp_out", input.number, 1, 1, 3 * embedDim);
        if(qkNorm && this.qNorm.getOutput() != null) {
        	this.qNorm.getOutput().viewOrg();
        	this.kNorm.getOutput().viewOrg();
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
        	this.dqt = CUDAMemoryManager.getCache("cache_dqt", batchSize, headNum, time, dk);
        	this.dkt = CUDAMemoryManager.getCache("cache_dkt", batchSize, headNum, time, dk);
        	this.dvt = CUDAMemoryManager.getCache("cache_dvt", batchSize, headNum, time, dk);
        	this.qkvDelta = CUDAMemoryManager.getCache("cache_qkv_delta", batchSize * time, 1, 1, 3 * embedDim);
        	if(!this.network.CUDNN_SDPA) {
        		this.dattn = CUDAMemoryManager.getCache("cache_dattn", batchSize, headNum, time, time);
        	}
//            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
//            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
//            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
//            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
//            int meryBackCount = dattn.dataLength + dqt.dataLength + dkt.dataLength + dvt.dataLength;
//            System.err.println(meryBackCount*4/1024/1024+"mb.");
        } else {
        	this.dqt.viewOrg(batchSize, headNum, time, dk);
        	this.dkt.viewOrg(batchSize, headNum, time, dk);
        	this.dvt.viewOrg(batchSize, headNum, time, dk);
        	this.qkvDelta.viewOrg(batchSize * time, 1, 1, 3 * embedDim);
        	if(!this.network.CUDNN_SDPA) {
        		this.dattn.viewOrg(batchSize, headNum, time, time);
        	}
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    private void projectQKV(Tensor input) {
        this.getqkvLinerLayer().forward(input);
        qkvFusedKernel.splitPermute(this.getqkvLinerLayer().getOutput(), qt, kt, vt, batchSize, time, headNum, dk);
    }

    private void projectQKVEval(Tensor input) {
        this.getqkvLinerLayer().forward(input, temp_out);
        qkvFusedKernel.splitPermute(temp_out, qt, kt, vt, batchSize, time, headNum, dk);
        temp_out.viewOrg();
    }

    private void backQKV(Tensor dqtDelta, Tensor dktDelta) {
        qkvFusedKernel.mergeUnpermute(dqtDelta, dktDelta, dvt, qkvDelta, batchSize, time, headNum, dk);
        this.getqkvLinerLayer().getOutput().viewOrg();
        this.getqkvLinerLayer().back(qkvDelta, this.getqkvLinerLayer().getOutput());
        this.diff = this.getqkvLinerLayer().diff;
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub

        projectQKV(input);
        
   	 	if(qkNorm) {
        	qNorm.forward(qt);
        	kNorm.forward(kt);
        	scaledDotProductAttention(qNorm.getOutput(), kNorm.getOutput(), vt);
        }else {
        	scaledDotProductAttention(qt, kt, vt);
        }
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        projectQKV(input);

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
        ropeKernel.forward2d(cos, sin, q, rq, time, headNum, dk);
        ropeKernel.forward2d(cos, sin, k, rk, time, headNum, dk);

        scaledDotProductAttention(rq, rk, vt);
        
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output(Tensor cos,Tensor sin, int igone) {
        // TODO Auto-generated method stub
        projectQKV(input);
        
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
        ropeKernel.forward2d(cos, sin, q, rq, time, headNum, dk, igone);
        ropeKernel.forward2d(cos, sin, k, rk, time, headNum, dk, igone);

        if(network.CUDNN_SDPA) {
        	cudnn_sdpa.forward(rq, rk, vt, temp);
        }else {
        	scaledDotProductAttention(rq, rk, vt);
        }

        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output(Tensor cos, Tensor sin, Tensor idskeep, int igone) {
        // TODO Auto-generated method stub
        projectQKV(input);
        
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
        ropeKernel.forward2d(cos, sin, idskeep, q, rq, time, headNum, dk, igone);
        ropeKernel.forward2d(cos, sin, idskeep, k, rk, time, headNum, dk, igone);

//        rq.showDM("rq");
//        rq.showDMByOffsetRed(77 * dk, dk, "rq");
        scaledDotProductAttention(rq, rk, vt);
        
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output_txt(Tensor cos, Tensor sin, Tensor idskeep, int txt_len) {
        // TODO Auto-generated method stub
        projectQKV(input);
        
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
        ropeKernel.forward2d_txt(cos, sin, idskeep, q, rq, time, headNum, dk, txt_len);
        ropeKernel.forward2d_txt(cos, sin, idskeep, k, rk, time, headNum, dk, txt_len);

//        rq.showDM("rq");
//        rq.showDMByOffsetRed(77 * dk, dk, "rq");
        scaledDotProductAttention(rq, rk, vt);
        
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi);
        this.output = this.getoLinerLayer().getOutput();

    }
    
    public void output_eval(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        projectQKVEval(input);

        ropeKernel.forward2d(cos, sin, qt, rq, time, headNum, dk);
        ropeKernel.forward2d(cos, sin, kt, rk, time, headNum, dk);
        
        
        if(qkNorm) {
        	qNorm.forward(rq, rq);
        	kNorm.forward(rk, rk);
        }
        scaledDotProductAttention(rq, rk, vt);
        Tensor vaccum = temp;
        attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

        this.getoLinerLayer().forward(oi, output);
        this.output = this.getoLinerLayer().getOutput();

    }

    public void output_eval(Tensor cos,Tensor sin, int igone) {
        // TODO Auto-generated method stub
    	 projectQKVEval(input);

         Tensor q = qt;
         Tensor k = kt;
         if(qkNorm) {
        	 qNorm.forward(qt);
        	 kNorm.forward(kt);
        	 q = qNorm.getOutput();
        	 k = kNorm.getOutput();
         }
         
         ropeKernel.forward2d(cos, sin, q, rq, time, headNum, dk, igone);
         ropeKernel.forward2d(cos, sin, k, rk, time, headNum, dk, igone);

         if(network.CUDNN_SDPA) {
         	cudnn_sdpa.forward(rq, rk, vt, temp);
         }else {
         	scaledDotProductAttention(rq, rk, vt);
         }
         
         Tensor vaccum = temp;
         attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

         this.getoLinerLayer().forward(oi, output);
         this.output = this.getoLinerLayer().getOutput();
    }
    
    public void output_eval(Tensor cos, Tensor sin, Tensor idskeep, int igone) {
        // TODO Auto-generated method stub
    	 projectQKVEval(input);

         Tensor q = qt;
         Tensor k = kt;
         if(qkNorm) {
        	 qNorm.forward(qt);
        	 kNorm.forward(kt);
        	 q = qNorm.getOutput();
        	 k = kNorm.getOutput();
         }
         
         ropeKernel.forward2d(cos, sin, idskeep, q, rq, time, headNum, dk, igone);
         ropeKernel.forward2d(cos, sin, idskeep, k, rk, time, headNum, dk, igone);

         scaledDotProductAttention(rq, rk, vt);
         Tensor vaccum = temp;
         attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);

         this.getoLinerLayer().forward(oi, output);
         this.output = this.getoLinerLayer().getOutput();
    }
    
    public void scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
//    	JCuda.cudaDeviceSynchronize();
//    	long start = System.nanoTime();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor preatt = temp;
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
        Tensor_OP().mul(preatt, d_k, preatt);
        softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);
        Tensor tmp = attn;

        Tensor vaccum = temp;
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);

//        JCuda.cudaDeviceSynchronize();
//        System.out.println((System.nanoTime() - start)/1e6+"ms.forward");
    }

    public void scaledDotProductAttentionBackward(Tensor q, Tensor k) {
//    	JCuda.cudaDeviceSynchronize();
//    	long start = System.nanoTime();
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
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);

        // backward into preatt
        softmaxKernel.softmax_backward(attn, dattn, dattn);

        //		dattn.showDM();
        float d_k = (float) (1.0f / Math.sqrt(dk));
        Tensor_OP().mul(dattn, d_k, dattn);
        Tensor dpreatt = dattn;
        
        /**
         * backward into dqt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, k.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
//        dqt.showDM("---");
        /**
         * backward into dkt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, q.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
       
//        JCuda.cudaDeviceSynchronize();
//        System.out.println((System.nanoTime() - start)/1e6+"ms.backward");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.getoLinerLayer().back(delta, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward(qt, kt);
        Tensor dqt_d = dqt;
        Tensor dkt_d = dkt;
        if(qkNorm) {
        	qNorm.back(dqt);
        	kNorm.back(dkt);
        	dqt_d = qNorm.diff;
            dkt_d = kNorm.diff;
        }
        backQKV(dqt_d, dkt_d);
    }
    
    public void diff(Tensor cos, Tensor sin) {
    	 this.getoLinerLayer().back(delta, oi);
         attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
         scaledDotProductAttentionBackward(rq, rk);
         /**
          * RoPE backward
          */
         ropeKernel.backward2d(cos, sin, dqt, rq, time, headNum, dk);
         ropeKernel.backward2d(cos, sin, dkt, rk, time, headNum, dk);

         Tensor dqt_d = rq;
         Tensor dkt_d = rk;
        
         if(qkNorm) {
          	qNorm.back(dqt_d);
          	kNorm.back(dkt_d);
          	dqt_d = qNorm.diff;
          	dkt_d = kNorm.diff;
         }
         
      	 backQKV(dqt_d, dkt_d);
    }
    
    public void diff(Tensor cos, Tensor sin, int igone) {
    	this.getoLinerLayer().back(delta, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        
        if(network.CUDNN_SDPA) {
        	cudnn_sdpa.backward(temp, dqt, dkt, dvt);
        }else {
        	scaledDotProductAttentionBackward(rq, rk);
        }

//        dqt.showDM("dqt");
        /**
         * RoPE backward
         */
        ropeKernel.backward2d(cos, sin, dqt, rq, time, headNum, dk, igone);
        ropeKernel.backward2d(cos, sin, dkt, rk, time, headNum, dk, igone);

        Tensor dqt_d = rq;
        Tensor dkt_d = rk;
       
        if(qkNorm) {
         	qNorm.back(dqt_d, dqt);
         	kNorm.back(dkt_d, dkt);
         	dqt_d = qNorm.diff;
         	dkt_d = kNorm.diff;
        }
        
     	backQKV(dqt_d, dkt_d);
    }
    
    public void diff(Tensor cos, Tensor sin, Tensor idskeep, int igone) {
    	this.getoLinerLayer().back(delta, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward(rq, rk);
        
//        dqt.showDM("dqt");
        /**
         * RoPE backward
         */
        ropeKernel.backward2d(cos, sin, idskeep, dqt, rq, time, headNum, dk, igone);
        ropeKernel.backward2d(cos, sin, idskeep, dkt, rk, time, headNum, dk, igone);

        Tensor dqt_d = rq;
        Tensor dkt_d = rk;
       
        if(qkNorm) {
         	qNorm.back(dqt_d, dqt);
         	kNorm.back(dkt_d, dkt);
         	dqt_d = qNorm.diff;
         	dkt_d = kNorm.diff;
        }
        
     	backQKV(dqt_d, dkt_d);
    }
    
    public void diff_txt(Tensor cos, Tensor sin, Tensor idskeep, int txt_len) {
    	this.getoLinerLayer().back(delta, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward(rq, rk);

        /**
         * RoPE backward
         */
        ropeKernel.backward2d_txt(cos, sin, idskeep, dqt, rq, time, headNum, dk, txt_len);
        ropeKernel.backward2d_txt(cos, sin, idskeep, dkt, rk, time, headNum, dk, txt_len);

        Tensor dqt_d = rq;
        Tensor dkt_d = rk;
       
        if(qkNorm) {
         	qNorm.back(dqt_d, dqt);
         	kNorm.back(dkt_d, dkt);
         	dqt_d = qNorm.diff;
         	dkt_d = kNorm.diff;
        }
        
     	backQKV(dqt_d, dkt_d);
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
    
    public void forward(Tensor input,Tensor cos,Tensor sin) {
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
            this.output_eval(cos, sin);
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
            this.output(cos, sin);
    	}  
    }
    
    public void forward(Tensor input,Tensor cos,Tensor sin, int igone) {
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
            this.output_eval(cos, sin, igone);
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
            this.output(cos, sin, igone);
    	}  
    }
    
    public void forward(Tensor input,Tensor cos,Tensor sin, Tensor idskeep, int igone) {
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
            this.output_eval(cos, sin, idskeep, igone);
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
            this.output(cos, sin, idskeep, igone);
    	}  
    }
    
    public void forward_txt(Tensor input,Tensor cos,Tensor sin, Tensor idskeep, int txt_len) {
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
        this.output_txt(cos, sin, idskeep, txt_len);
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
    
    public void back(Tensor delta,Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos, sin);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    public void back(Tensor delta,Tensor cos,Tensor sin,int igone) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos, sin, igone);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    public void back(Tensor delta,Tensor cos,Tensor sin, Tensor idskeep, int igone) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos, sin, idskeep, igone);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    public void back_txt(Tensor delta,Tensor cos,Tensor sin, Tensor idskeep, int txt_len) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff_txt(cos, sin, idskeep, txt_len);
    }
    
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	if(qkNorm) {
	        qNorm.update();
	        kNorm.update();
    	}
        getqkvLinerLayer().update();
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
    	if(qkNorm) {
	        qNorm.saveModel(outputStream);
	        kNorm.saveModel(outputStream);
    	}
        getqkvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(qkNorm) {
	        qNorm.loadModel(inputStream, 1, 1, dk, BNType.fully_bn);
	        kNorm.loadModel(inputStream, 1, 1, dk, BNType.fully_bn);
    	}
        getqkvLinerLayer().loadModel(inputStream);
        getoLinerLayer().loadModel(inputStream);
    }

    public FullyLayer getqkvLinerLayer() {
        return qkvLinerLayer;
    }

    public void setqkvLinerLayer(FullyLayer qkvLinerLayer) {
        this.qkvLinerLayer = qkvLinerLayer;
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
    	}
        qkvLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}

