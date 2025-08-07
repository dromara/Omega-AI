package com.omega.engine.nn.layer.dit.mmdit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTAttentionLayer2
 *
 * @author Administrator
 */
public class DiTJoinBlockHead extends Layer {
	
    private SiLULayer modulationAct;
    public FullyLayer adaLN_modulation;
	
//    public RMSLayer qNorm;
//    public RMSLayer kNorm;
//	
//    public RMSLayer norm1;
//    public RMSLayer norm2;

    public LNLayer qNorm;
    public LNLayer kNorm;
	
    public LNLayer norm1;
    public LNLayer norm2;
    
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    
    public MMDiTMLPLayer mlp;
    
    private int mlp_ratio;
    private int time;
    private int embedDim = 0;
    private int cEmbedDim;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    public Tensor shift_msa;
    public Tensor scale_msa;
    public Tensor gate_msa;
    public Tensor shift_mlp;
    public Tensor scale_mlp;
    public Tensor gate_mlp;

    private Tensor attnInput;
    private Tensor tmpOut;
    private Tensor mlpInput;

    private int batchSize = 1;
    private int n_mods = 6;
    
    private boolean qkNorm = false;
    
    private boolean pre_only = false;
    
    private int[] shape;

    public DiTJoinBlockHead(int embedDim, int cEmbedDim, int mlp_ratio, int time, boolean bias, boolean qkNorm, boolean pre_only) {
    	this.pre_only = pre_only;
        if(pre_only) {
        	n_mods = 2;
        }
        this.mlp_ratio = mlp_ratio;
        this.bias = bias;
        this.time = time;
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.qkNorm = qkNorm;
        this.bias = bias;
        this.channel = time;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.initLayers();
    }

    public DiTJoinBlockHead(int embedDim, int cEmbedDim, int mlp_ratio, int time, boolean bias, boolean qkNorm, boolean pre_only, Network network) {
    	this.pre_only = pre_only;
        if(pre_only) {
        	n_mods = 2;
        }
        this.mlp_ratio = mlp_ratio;
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.qkNorm = qkNorm;
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
       
//    	if(qkNorm) {
//        	qNorm = new RMSLayer(network);
//        	kNorm = new RMSLayer(network);
//        }
//    	
//    	this.norm1 = new RMSLayer(network);
//    	this.norm2 = new RMSLayer(network);
    	
    	if(qkNorm) {
	    	qNorm = new LNLayer(1, 1, embedDim, BNType.fully_bn, network);
	    	kNorm = new LNLayer(1, 1, embedDim, BNType.fully_bn, network);
	    }
    	
    	this.norm1 = new LNLayer(1, 1, embedDim, BNType.fully_bn, network);
    	
        this.modulationAct = new SiLULayer(network);
        
        this.adaLN_modulation = new FullyLayer(cEmbedDim, embedDim * n_mods, true, network);
//        this.adaLN_modulation.weight.clearGPU();
//        this.adaLN_modulation.bias.clearGPU();

        this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.qLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.qLinerLayer.bias != null) {
        	this.qLinerLayer.bias.clearGPU();
        }

        this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.kLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.kLinerLayer.bias != null) {
        	this.kLinerLayer.bias.clearGPU();
        }

        this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
        this.vLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
        if(this.vLinerLayer.bias != null) {
        	this.vLinerLayer.bias.clearGPU();
        }
        
        if(!pre_only) {
        	
        	this.norm2 = new LNLayer(1, 1, embedDim, BNType.fully_bn, network);
        	
	        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, true, this.network));
	        this.oLinerLayer.weight.setData(RandomUtils.xavierUniform(this.embedDim * this.embedDim, this.embedDim, this.embedDim,  1.0f));
	        if(this.oLinerLayer.bias != null) {
	        	this.oLinerLayer.bias.clearGPU();
	        }
        }
        
        if(!pre_only) {
        	int mlp_hidden_dim = embedDim * mlp_ratio;
        	mlp = new MMDiTMLPLayer(embedDim, mlp_hidden_dim, true, network);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number / time;

        if (this.output != null) {
            this.output.viewOrg();
        }

        if (this.output == null || this.output.number != this.batchSize) {
        	shape = new int[] {batchSize, n_mods, 1, embedDim};
        	shift_msa = Tensor.createGPUTensor(this.shift_msa, batchSize, 1, 1, embedDim, true);
        	scale_msa = Tensor.createGPUTensor(this.scale_msa, batchSize, 1, 1, embedDim, true);
        	if(!pre_only) {
        		gate_msa = Tensor.createGPUTensor(this.gate_msa, batchSize, 1, 1, embedDim, true);
        		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, batchSize, 1, 1, embedDim, true);
            	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, batchSize, 1, 1, embedDim, true);
            	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, batchSize, 1, 1, embedDim, true);
        	}
        	tmpOut = Tensor.createGPUTensor(this.attnInput, batchSize * time, 1, 1, embedDim, true);
        	mlpInput = Tensor.createGPUTensor(this.mlpInput, batchSize * time, 1, 1, embedDim, true);
        	attnInput = Tensor.createGPUTensor(this.attnInput, batchSize * time, 1, 1, embedDim, true);
            // [batch_size, len_q, n_heads * dim_v]
        	if(!pre_only) {
        		this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
        	}
        }
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
            if(!pre_only) {
            	this.getoLinerLayer().getOutput().viewOrg();
            }
        }
    }
    
//    public void init_eval(Tensor input) {
//        // TODO Auto-generated method stub
//        this.number = input.number;
//        this.batchSize = this.number/time;
//
//    	this.qt = CUDAMemoryManager.getCache("dit_block_attn_qt", batchSize, headNum, time, dk);
//    	this.kt = CUDAMemoryManager.getCache("dit_block_attn_kt", batchSize, headNum, time, dk);
//    	this.vt = CUDAMemoryManager.getCache("dit_block_attn_vt", batchSize, headNum, time, dk);
//
//        // [batch_size, len_q, n_heads * dim_v]
//        this.oi = CUDAMemoryManager.getCache("dit_block_attn_oi", batchSize * time, 1, 1, embedDim);
//        this.output = CUDAMemoryManager.getCache("dit_block_attn_out", input.number, input.channel, input.height, input.width);
//    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
//        if (this.dattn == null) {
//            this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
//            this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
//            this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
//            this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
//        } else {
//            dattn.viewOrg();
//            dqt.viewOrg();
//            dkt.viewOrg();
//            dvt.viewOrg();
//        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub

    }
    
    public void modulate(Tensor x,Tensor shift,Tensor scale,Tensor output) {
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(scale, 1.0f, scale);
    	Tensor_OP().mul(x, scale, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().addAxis(output, shift, output, batchSize, time, 1, output.width, 1);
    }
    
    public void pre_attention(Tensor input, Tensor c) {
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 设置输入
         */
        this.setInput(input);
    	
    	modulationAct.forward(c);

    	adaLN_modulation.forward(modulationAct.getOutput());
    	
    	int[] shape = new int[] {batchSize, n_mods, 1, embedDim};

    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_msa, shape, 0);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_msa, shape, 1);
    	if(!pre_only) {
	    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
	    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_mlp, shape, 3);
	    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_mlp, shape, 4);
	    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	}
    	
//    	input.showDMByOffsetRed(10 * 384, 384, "sdfsdf");
//    	norm1.gamma.showDM("gammae");
//    	norm1.beta.showDM("beta");
        norm1.forward(input);

//        norm1.getOutput().showDM("norm1");
    	modulate(norm1.getOutput(), shift_msa, scale_msa, attnInput);
    	
        Tensor x = attnInput;
//    	x.showDM("q_in");
        this.getqLinerLayer().forward(x);
        this.getkLinerLayer().forward(x);
        this.getvLinerLayer().forward(x);
        
//        this.getqLinerLayer().getOutput().showDM("q");

    }
    
    public void pre_attention_back(Tensor xDiff, Tensor dc) {
    	Tensor dq =  this.getqLinerLayer().getOutput().view(batchSize * time, 1, 1, width);
    	Tensor dk =  this.getkLinerLayer().getOutput().view(batchSize * time, 1, 1, width);
    	Tensor dv =  this.getvLinerLayer().getOutput().view(batchSize * time, 1, 1, width);
    	
    	qLinerLayer.back(dq);
    	kLinerLayer.back(dk);
    	vLinerLayer.back(dv);
    	
    	Tensor dattnInput = qLinerLayer.diff;
    	Tensor_OP().add(dattnInput, kLinerLayer.diff, dattnInput);
    	Tensor_OP().add(dattnInput, vLinerLayer.diff, dattnInput);
    	
    	Tensor tmp = shift_msa;
    	Tensor x = norm1.getOutput();
    	Tensor scale = scale_msa;
    	modulate_back2(tmp, norm1.getOutput(), x, scale, dattnInput, 0, 1);

    	adaLN_modulation.back(adaLN_modulation.getOutput());

    	modulationAct.back(adaLN_modulation.diff);
    	
    	Tensor_OP().add(dc, modulationAct.diff, dc);

    	norm1.back(norm1.getOutput());
    	
    	Tensor_OP().add(norm1.diff, xDiff, norm1.diff);
    	
    	this.diff = norm1.diff;
    }
    
    public void pre_attention_back(Tensor xDiff, Tensor dc, Tensor drq, Tensor drk) {
    	Tensor dq =  drq.view(batchSize * time, 1, 1, width);
    	Tensor dk =  drk.view(batchSize * time, 1, 1, width);
    	Tensor dv =  this.getvLinerLayer().getOutput().view(batchSize * time, 1, 1, width);
    	
    	qLinerLayer.back(dq);
    	kLinerLayer.back(dk);
    	vLinerLayer.back(dv);
    	
    	Tensor dattnInput = qLinerLayer.diff;
    	Tensor_OP().add(dattnInput, kLinerLayer.diff, dattnInput);
    	Tensor_OP().add(dattnInput, vLinerLayer.diff, dattnInput);
    	
    	Tensor tmp = shift_msa;
    	Tensor x = norm1.getOutput();
    	Tensor scale = scale_msa;
    	modulate_back2(tmp, norm1.getOutput(), x, scale, dattnInput, 0, 1);

    	adaLN_modulation.back(adaLN_modulation.getOutput());

    	modulationAct.back(adaLN_modulation.diff);
    	
    	Tensor_OP().add(dc, modulationAct.diff, dc);

    	norm1.back(norm1.getOutput());
    	
    	Tensor_OP().add(norm1.diff, xDiff, norm1.diff);
    	
    	this.diff = norm1.diff;
    }
    
    public void post_attention(Tensor attn) {
    	
    	oLinerLayer.forward(attn);
    	
    	Tensor_OP().mul(oLinerLayer.getOutput(), gate_msa, tmpOut, batchSize, time, 1, tmpOut.width, 1);
    	Tensor_OP().add(input, tmpOut, tmpOut);
    	
    	norm2.forward(tmpOut);

    	modulate(norm2.getOutput(), shift_mlp, scale_mlp, mlpInput);

    	mlp.forward(mlpInput);

    	Tensor_OP().mul(mlp.getOutput(), gate_mlp, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().add(tmpOut, output, output);
    	
    }
    
//    public void modulate_back(Tensor dShift,Tensor dScale,Tensor dx,Tensor x,Tensor scale,Tensor delta) {
//    	Tensor_OP().addAxisBack(dShift, delta, batchSize, time, 1, delta.width, 1);
//    	Tensor_OP().mul_right_back(x, delta, dScale, batchSize, time, 1, delta.width, 1);
//    	Tensor_OP().mul_left_back(scale, delta, dx,  batchSize, time, 1, delta.width, 1);
//    }
    
    public void modulate_back2(Tensor tmp,Tensor dx,Tensor x,Tensor scale,Tensor delta, int idx1, int idx2) {
    	Tensor_OP().addAxisBack(tmp, delta, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), tmp, shape, idx1);
    	Tensor_OP().mul_right_back(x, delta, tmp, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), tmp, shape, idx2);
    	Tensor_OP().mul_left_back(scale, delta, dx,  batchSize, time, 1, delta.width, 1);
    }
    
    public Tensor post_attention_back(Tensor delta, String name) {
//    	adaLN_modulation.getOutput().clearGPU();
    	Tensor_OP().mul_left_back(gate_mlp, delta, output,  batchSize, time, 1, embedDim, 1);

    	mlp.back(output);

    	Tensor_OP().mul_right_back(mlp.getOutput(), delta, gate_mlp, batchSize, time, 1, embedDim, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	
    	Tensor tmp = shift_mlp;
    	Tensor x = norm2.getOutput();
    	Tensor scale = scale_mlp;
    	modulate_back2(tmp, output, x, scale, mlp.diff, 3, 4);

    	norm2.back(output);
    	Tensor_OP().add(norm2.diff, delta, norm2.diff);
    	
    	Tensor_OP().mul_left_back(gate_msa, norm2.diff, delta,  batchSize, time, 1, embedDim, 1);

    	oLinerLayer.back(delta);
    	
    	Tensor_OP().mul_right_back(oLinerLayer.getOutput(), norm2.diff, gate_msa, batchSize, time, 1, embedDim, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
    	
    	return norm2.diff;
    }
    
    public Tensor q() {
    	return qLinerLayer.getOutput().view(batchSize, time, 1, embedDim);
    }
    
    public Tensor k() {
    	return kLinerLayer.getOutput().view(batchSize, time, 1, embedDim);
    }
    
    public Tensor v() {
    	return vLinerLayer.getOutput().view(batchSize, time, 1, embedDim);
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
    	if(qkNorm) {
        	qNorm.update();
        	kNorm.update();
        }
    	
    	this.norm1.update();
    	
        this.adaLN_modulation.update();

        this.qLinerLayer.update();
        this.kLinerLayer.update();
        this.vLinerLayer.update();
        
        if(!pre_only) {
        	this.norm2.update();
	        this.oLinerLayer.update();
	        mlp.update();
        }
        
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
    	norm1.saveModel(outputStream);
    	this.adaLN_modulation.saveModel(outputStream);
        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        if(!pre_only) {
        	this.norm2.saveModel(outputStream);
	        this.oLinerLayer.saveModel(outputStream);
	        mlp.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(qkNorm) {
	        qNorm.loadModel(inputStream, 1, 1, width, BNType.fully_bn);
	        kNorm.loadModel(inputStream, 1, 1, width, BNType.fully_bn);
    	}
    	norm1.loadModel(inputStream, 1, 1, width, BNType.fully_bn);
    	this.adaLN_modulation.loadModel(inputStream);
        getqLinerLayer().loadModel(inputStream);
        getkLinerLayer().loadModel(inputStream);
        getvLinerLayer().loadModel(inputStream);
        if(!pre_only) {
        	this.norm2.loadModel(inputStream, 1, 1, width, BNType.fully_bn);
	        this.oLinerLayer.loadModel(inputStream);
	        mlp.loadModel(inputStream);
        }
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

