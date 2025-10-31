package com.omega.engine.nn.layer.dit.sana;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * Lightweight linear attention
 *
 * @author Administrator
 */
public class LiteLALayer extends Layer {
	
    private int batchSize = 1;
    private int time;
    private int headNum = 1;
    private int embedDim = 0;
    private int dk = 0;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;
    
    private boolean qkNorm = false;
    
    public RMSLayer qNorm;
    public RMSLayer kNorm;
    
    public FullyLayer qLinear;
    public FullyLayer kLinear;
    public FullyLayer vLinear;
    public FullyLayer oLinear;
    
    private ReluLayer q_a;
    private ReluLayer k_a;

    private Tensor qt;
    private Tensor kt;
    private Tensor vt;
    
    private Tensor v_1;
    private Tensor vk;
    private Tensor out;
    private Tensor out_;
    private Tensor oi;
    
    private Tensor dq_a;
    private Tensor dk_a;
    
    private int[] p_0231 = new int[]{0, 2, 3, 1};
    
    private AttentionKernel attentionKernel;
    private PaddingKernel paddingKernel;
    
    public LiteLALayer(int embedDim, int headNum, int time, boolean qkNorm, boolean bias) {
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

    public LiteLALayer(int embedDim, int headNum, int time, boolean qkNorm, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.bias = bias;
        this.qkNorm = qkNorm;
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
    	
    	if(qkNorm) {
    		qNorm = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    		kNorm = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    	}
    	
		this.qLinear = new FullyLayer(embedDim, embedDim, bias, this.network);
		RandomUtils.xavier_uniform(this.qLinear.weight, 1, embedDim, embedDim);
		if(this.qLinear.bias != null) {
			this.qLinear.bias.clearGPU();
		}
		this.kLinear = new FullyLayer(embedDim, embedDim, bias, this.network);
		RandomUtils.xavier_uniform(this.kLinear.weight, 1, embedDim, embedDim);
		if(this.kLinear.bias != null) {
			this.kLinear.bias.clearGPU();
		}
		this.vLinear = new FullyLayer(embedDim, embedDim, bias, this.network);
		RandomUtils.xavier_uniform(this.vLinear.weight, 1, embedDim, embedDim);
		if(this.vLinear.bias != null) {
			this.vLinear.bias.clearGPU();
		}
		this.oLinear = new FullyLayer(embedDim, embedDim, true, this.network);
		RandomUtils.xavier_uniform(this.oLinear.weight, 1, embedDim, embedDim);
		if(this.oLinear.bias != null) {
			this.oLinear.bias.clearGPU();
		}
		
		q_a = new ReluLayer(network);
		k_a = new ReluLayer(network);

        if (attentionKernel == null) {
            attentionKernel = new AttentionKernel(cuda());
        }
        
		if(paddingKernel == null) {
			paddingKernel = new PaddingKernel(this.cuda());
		}
		
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
    	this.number = input.number;
    	this.batchSize = number / time;
    	
    	 if (this.qt != null) {
             this.output.viewOrg();
             this.qt.viewOrg();
             this.kt.viewOrg();
             this.vt.viewOrg();
         }
         if (this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.dk) {
             // [batch_size，time，head_num，d_k]
             this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, dk, time, true);
             this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, dk, time, true);
             this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, dk, time, true);
             this.v_1 = Tensor.createGPUTensor(this.v_1, batchSize, headNum, dk + 1, time, true);
             this.vk = Tensor.createGPUTensor(this.vk, batchSize, headNum, dk + 1, dk, true);
             this.out = Tensor.createGPUTensor(this.out, batchSize, headNum, dk + 1, time, true);
             this.out_ = Tensor.createGPUTensor(this.out_, batchSize, headNum, dk, time, true);
             this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
         }
         if (this.qLinear.getOutput() != null) {
             this.qLinear.getOutput().viewOrg();
             this.kLinear.getOutput().viewOrg();
             this.vLinear.getOutput().viewOrg();
             this.oLinear.getOutput().viewOrg();
         }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	 if (this.dq_a == null) {
         	this.dq_a = CUDAMemoryManager.getCache("cache_dqt", batchSize, headNum, dk, time);
         	this.dk_a = CUDAMemoryManager.getCache("cache_dkt", batchSize, headNum, dk, time);
         } else {
         	this.dq_a.viewOrg(batchSize, headNum, dk, time);
         	this.dk_a.viewOrg(batchSize, headNum, dk, time);
         }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	
    	qLinear.forward(input);
    	kLinear.forward(input);
    	vLinear.forward(input);
    	
    	Tensor q = qLinear.getOutput();
    	Tensor k = kLinear.getOutput();
    	if(qkNorm) {
    		qNorm.forward(qLinear.getOutput());
    		kNorm.forward(kLinear.getOutput());
    		q = qNorm.getOutput();
        	k = kNorm.getOutput();
    	}
    	
        Tensor query = q.view(batchSize, time, headNum, dk);
        Tensor key = k.view(batchSize, time, headNum, dk);
        Tensor value = vLinear.getOutput().view(batchSize, time, headNum, dk);
    	
        Tensor_OP().permute(query, qt, p_0231);
        Tensor_OP().permute(key, kt, p_0231);
        Tensor_OP().permute(value, vt, p_0231);

        q_a.forward(qt);
        k_a.forward(kt);

        paddingKernel.padding2d(vt, v_1, new int[] {0, 0, 0, 1}, 1);
        
        GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, dk, dk+1, time, 1.0f, k_a.getOutput().getGpuData(), time, dk * time, v_1.getGpuData(), time, (dk + 1) * time, 0.0f, vk.getGpuData(), dk, (dk+1) * dk, batchSize * headNum);

        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, time, dk+1, dk, 1.0f, q_a.getOutput().getGpuData(), time, dk * time, vk.getGpuData(), dk, (dk+1) * dk, 0.0f, out.getGpuData(), time, (dk+1) * time, batchSize * headNum);

        // out_ = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        attentionKernel.outDivLastDim(out, out_);
        
        Tensor_OP().permute(out_, oi, new int[] {batchSize, headNum, dk, time}, new int[] {batchSize, time, headNum, dk}, new int[] {0, 3, 1, 2});

        oLinear.forward(oi);
        
        this.output = oLinear.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	oLinear.back(delta, oi);
    	
    	Tensor_OP().permute(oi, out_, new int[] {batchSize, time, headNum, dk}, new int[] {batchSize, headNum, dk, time}, p_0231);
    	
    	//out_ = out[:, :, :-1] / (out[:, :, -1:] + self.eps) backward
    	attentionKernel.outDivLastDimBack(out, out_);
    	
    	GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, time, dk, dk+1, 1.0f, out.getGpuData(), time, (dk+1) * time, vk.getGpuData(), dk, (dk+1) * dk, 0.0f, dq_a.getGpuData(), time, dk * time, batchSize * headNum);
    	GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, dk, dk+1, time, 1.0f, q_a.getOutput().getGpuData(), time, dk * time, out.getGpuData(), time, (dk+1) * time, 0.0f, vk.getGpuData(), dk, (dk+1) * dk, batchSize * headNum);

        /**
         * backward into dkt
         */
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, time, dk, dk+1, 1.0f, v_1.getGpuData(), time, (dk+1) * time, vk.getGpuData(), dk, (dk+1) * dk, 0.0f, dk_a.getGpuData(), time, dk * time, batchSize * headNum);
  
        GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, time, dk+1, dk, 1.0f, k_a.getOutput().getGpuData(), time, dk * time, vk.getGpuData(), dk, (dk+1) * dk, 0.0f, v_1.getGpuData(), time, (dk+1) * time, batchSize * headNum);
    	
    	paddingKernel.padding2dGrad(v_1, vt, new int[] {0, 0, 0, 1});
    	
    	k_a.back(dk_a);
    	q_a.back(dq_a);
    	
    	Tensor qDelta = qLinear.getOutput().view(batchSize, time, headNum, dk);
    	Tensor kDelta = kLinear.getOutput().view(batchSize, time, headNum, dk);
    	Tensor vDelta = vLinear.getOutput().view(batchSize, time, headNum, dk);

    	Tensor_OP().permute(q_a.diff, qDelta, new int[] {0, 3, 1, 2});
        Tensor_OP().permute(k_a.diff, kDelta, new int[] {0, 3, 1, 2});
        Tensor_OP().permute(vt, vDelta, new int[] {0, 3, 1, 2});
    	
        qDelta = qDelta.view(batchSize * time, 1, 1, headNum * dk);
        kDelta = kDelta.view(batchSize * time, 1, 1, headNum * dk);
        vDelta = vDelta.view(batchSize * time, 1, 1, headNum * dk);
        
        if(qkNorm) {
        	qNorm.back(qDelta);
        	kNorm.back(kDelta);
        	qDelta = qNorm.diff;
        	kDelta = kNorm.diff;
        }
        
        qt.view(input.shape());
        kt.view(input.shape());
        vt.view(input.shape());
        qLinear.back(qDelta, qt);
        kLinear.back(kDelta, kt);
        vLinear.back(vDelta, vt);
        Tensor_OP().add(qLinear.diff, kLinear.diff, qLinear.diff);
        Tensor_OP().add(qLinear.diff, vLinear.diff, qLinear.diff);
        // dxt
        this.diff = qLinear.diff;
    }
    
    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 参数初始化
         */
        this.init();
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
        this.setDelta();
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
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init(input);
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
    	qLinear.update();
    	kLinear.update();
    	vLinear.update();
    	oLinear.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.linearAttn;
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
    	if(qkNorm) {
    		qNorm.saveModel(outputStream);
    		kNorm.saveModel(outputStream);
    	}
    	qLinear.saveModel(outputStream);
    	kLinear.saveModel(outputStream);
    	vLinear.saveModel(outputStream);
    	oLinear.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(qkNorm) {
    		qNorm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
    		kNorm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
    	}
    	qLinear.loadModel(inputStream);
    	kLinear.loadModel(inputStream);
    	vLinear.loadModel(inputStream);
    	oLinear.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	if(qkNorm) {
    		qNorm.accGrad(scale);
    		kNorm.accGrad(scale);
    	}
    	qLinear.accGrad(scale);
    	kLinear.accGrad(scale);
    	vLinear.accGrad(scale);
    	oLinear.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, LiteLALayer block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(block.qLinear.weight, weightMap, "q.weight");
        ModeLoaderlUtils.loadData(block.kLinear.weight, weightMap, "k.weight");
        ModeLoaderlUtils.loadData(block.vLinear.weight, weightMap, "v.weight");
        ModeLoaderlUtils.loadData(block.oLinear.weight, weightMap, "proj.weight");
        ModeLoaderlUtils.loadData(block.oLinear.bias, weightMap, "proj.bias");
    }
    
    public static void main(String[] args) {
    	
    	String inputPath = "D:\\models\\linearAttn_w.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	
        int batchSize = 2;
        int time = 16;
        int embedDim = 32;
        int headNum = 4;
        
        Transformer tf = new Transformer();
        tf.number = batchSize * time;
        tf.time = time;
        
        LiteLALayer lite = new LiteLALayer(embedDim, headNum, time, false, false, tf);
        
        loadWeight(datas, lite, true);
        
	    String xPath = "D:\\models\\linearAttn_x.json";
	    Map<String, Object> xDatas = LagJsonReader.readJsonFileSmallWeight(xPath);
	    Tensor input = new Tensor(batchSize, time, 1, embedDim, true);
	    ModeLoaderlUtils.loadData(input, xDatas, "x", 3);
	    input.view(batchSize * time, 1, 1, embedDim);
	    
	    String dxPath = "D:\\models\\linearAttn_dx.json";
	    Map<String, Object> dxDatas = LagJsonReader.readJsonFileSmallWeight(dxPath);
	    Tensor delta = new Tensor(batchSize, time, 1, embedDim, true);
	    ModeLoaderlUtils.loadData(delta, dxDatas, "dx", 3);
	    delta.view(batchSize * time, 1, 1, embedDim);
	    
        for (int i = 0; i < 10; i++) {
        	lite.forward(input);
        	lite.getOutput().showDM("output");
        	lite.back(delta);
        	lite.diff.showDM("dx");
        }
    }
}

