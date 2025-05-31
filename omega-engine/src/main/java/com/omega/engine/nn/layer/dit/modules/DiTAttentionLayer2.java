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
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTAttentionLayer2
 *
 * @author Administrator
 */
public class DiTAttentionLayer2 extends Layer {
	
    public LNLayer qNorm;
    public LNLayer kNorm;
	
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    //	public GNLayer gn;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    private AttentionKernel attentionKernel;
    private SoftmaxCudnnKernel softmaxKernel;
    private RoPEKernel ropeKernel;
    
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
    
    private Tensor temp_out;

    public DiTAttentionLayer2(int embedDim, int headNum, int time, boolean bias, boolean qkNorm) {
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

    public DiTAttentionLayer2(int embedDim, int headNum, int time, boolean bias, boolean qkNorm, Network network) {
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
        	qNorm = new LNLayer(network);
        	kNorm = new LNLayer(network);
        }
    	
        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.qLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.qLinerLayer.bias != null) {
        	this.qLinerLayer.bias.clearGPU();
        }
        //		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, MatrixUtils.order(embedDim * embedDim, 0.1f, 0.01f), true);
        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.kLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.kLinerLayer.bias != null) {
        	this.kLinerLayer.bias.clearGPU();
        }
        //		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, kDim, MatrixUtils.order(embedDim * kDim, 0.1f, 0.01f), true);
        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.vLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.vLinerLayer.bias != null) {
        	this.vLinerLayer.bias.clearGPU();
        }
        //		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, vDim, MatrixUtils.order(embedDim * vDim, 0.1f, 0.01f), true);
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.oLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
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
            //			JCuda.cudaDeviceSynchronize();
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
        	this.rq = Tensor.createGPUTensor(this.rq, batchSize, time, headNum, dk, true);
            this.rk = Tensor.createGPUTensor(this.rk, batchSize, time, headNum, dk, true);
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
            this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
            //			this.output.showShape("output");
        }
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
            this.getoLinerLayer().getOutput().viewOrg();
        }
    }
    
    public void init_eval(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number/time;
        this.rq = CUDAMemoryManager.getCache("dit_block_attn_rq", batchSize, time, headNum, dk);
    	this.rk = CUDAMemoryManager.getCache("dit_block_attn_rk", batchSize, time, headNum, dk);
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
        this.temp_out = CUDAMemoryManager.getCache("dit_block_attn_temp_out", input.number, input.channel, input.height, input.width);
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dattn == null) {
            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
        } else {
            dattn.viewOrg();
            dqt.viewOrg();
            dkt.viewOrg();
            dvt.viewOrg();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	
        this.getqLinerLayer().forward(input);
        this.getkLinerLayer().forward(input);
        this.getvLinerLayer().forward(input);

        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor_OP().permute(query, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(key, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        
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
        this.getqLinerLayer().forward(input);
        this.getkLinerLayer().forward(input);
        this.getvLinerLayer().forward(input);

        Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);
        /**
         * apply RoPE
         */
        ropeKernel.forward2d(cos, sin, query, rq, time, headNum, dk);
        ropeKernel.forward2d(cos, sin, key, rk, time, headNum, dk);
        
        Tensor_OP().permute(rq, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(rk, kt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});

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
    
    public void output_eval(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        this.getqLinerLayer().forward(input, temp_out);
        Tensor query = temp_out.view(batchSize, time, headNum, dk);
        ropeKernel.forward2d(cos, sin, query, rq, time, headNum, dk);
        
        temp_out.viewOrg();
        this.getkLinerLayer().forward(input, temp_out);
        Tensor key = temp_out.view(batchSize, time, headNum, dk);
        ropeKernel.forward2d(cos, sin, key, rk, time, headNum, dk);
        
        temp_out.viewOrg();
        this.getvLinerLayer().forward(input, temp_out);
        Tensor value = temp_out.view(batchSize, time, headNum, dk);
        Tensor_OP().permute(value, vt, new int[]{0, 2, 1, 3});
        temp_out.viewOrg();
        
        Tensor_OP().permute(rq, qt, new int[]{0, 2, 1, 3});
        Tensor_OP().permute(rk, kt, new int[]{0, 2, 1, 3});

        if(qkNorm) {
        	qNorm.forward(qt, qt);
        	kNorm.forward(kt, kt);
        }
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
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
        /**
         * backward into dkt

         */
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
        this.getoLinerLayer().back(delta, oi);
        attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
        scaledDotProductAttentionBackward();
        qt.view(this.getqLinerLayer().getOutput().shape());
        kt.view(this.getkLinerLayer().getOutput().shape());
        vt.view(this.getvLinerLayer().getOutput().shape());
        if(qkNorm) {
        	qNorm.back(dqt);
        	kNorm.back(dkt);
        	Tensor_OP().permute(qNorm.diff, qt, new int[]{0, 2, 1, 3});
            Tensor_OP().permute(kNorm.diff, kt, new int[]{0, 2, 1, 3});
        }else {
        	Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
            Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
        }
        Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
        Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
        Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
        this.getqLinerLayer().back(queryDelta);
        this.getkLinerLayer().back(keyDelta);
        this.getvLinerLayer().back(valueDelta);
        Tensor_OP().add(this.getqLinerLayer().diff, this.getkLinerLayer().diff, this.getqLinerLayer().diff);
        Tensor_OP().add(this.getqLinerLayer().diff, this.getvLinerLayer().diff, this.getqLinerLayer().diff);
        // dxt
        this.diff = this.getqLinerLayer().diff;
    }
    
    public void diff(Tensor cos, Tensor sin) {
    	 this.getoLinerLayer().back(delta, oi);
         attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
         scaledDotProductAttentionBackward();
         qt.view(this.getqLinerLayer().getOutput().shape());
         kt.view(this.getkLinerLayer().getOutput().shape());
         vt.view(this.getvLinerLayer().getOutput().shape());
         if(qkNorm) {
         	qNorm.back(dqt);
         	kNorm.back(dkt);
         	Tensor_OP().permute(qNorm.diff, qt, new int[]{0, 2, 1, 3});
            Tensor_OP().permute(kNorm.diff, kt, new int[]{0, 2, 1, 3});
         }else {
         	Tensor_OP().permute(dqt, qt, new int[]{0, 2, 1, 3});
            Tensor_OP().permute(dkt, kt, new int[]{0, 2, 1, 3});
         }
         Tensor_OP().permute(dvt, vt, new int[]{0, 2, 1, 3});
         /**
          * RoPE backward
          */
         ropeKernel.backward2d(cos, sin, qt, rq, time, headNum, dk);
         ropeKernel.backward2d(cos, sin, kt, rk, time, headNum, dk);
         Tensor queryDelta = rq.view(this.getqLinerLayer().getOutput().shape());
         Tensor keyDelta = rk.view(this.getkLinerLayer().getOutput().shape());
         Tensor valueDelta = vt.view(this.getvLinerLayer().getOutput().shape());
         this.getqLinerLayer().back(queryDelta);
         this.getkLinerLayer().back(keyDelta);
         this.getvLinerLayer().back(valueDelta);
         Tensor_OP().add(this.getqLinerLayer().diff, this.getkLinerLayer().diff, this.getqLinerLayer().diff);
         Tensor_OP().add(this.getqLinerLayer().diff, this.getvLinerLayer().diff, this.getqLinerLayer().diff);
         // dxt
         this.diff = this.getqLinerLayer().diff;
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

    @Override
    public void update() {
        // TODO Auto-generated method stub
    	if(qkNorm) {
	        qNorm.update();
	        kNorm.update();
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
    	if(qkNorm) {
	        qNorm.saveModel(outputStream);
	        kNorm.saveModel(outputStream);
    	}
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        getoLinerLayer().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(qkNorm) {
	        qNorm.loadModel(inputStream);
	        kNorm.loadModel(inputStream);
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
    	if(qkNorm) {
	        qNorm.accGrad(scale);
	        kNorm.accGrad(scale);
    	}
        qLinerLayer.accGrad(scale);
        kLinerLayer.accGrad(scale);
        vLinerLayer.accGrad(scale);
        oLinerLayer.accGrad(scale);
    }
}

