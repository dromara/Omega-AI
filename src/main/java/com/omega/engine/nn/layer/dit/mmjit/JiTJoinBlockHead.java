package com.omega.engine.nn.layer.dit.mmjit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.org.DiTSwiGLUFFN;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * JiTJoinBlockHead
 *
 * @author Administrator
 */
public class JiTJoinBlockHead extends Layer {

    public RMSLayer qNorm;
    public RMSLayer kNorm;
	
    public RMSLayer norm1;
    public RMSLayer norm2;
    
    public FullyLayer qLinerLayer;
    public FullyLayer kLinerLayer;
    public FullyLayer vLinerLayer;
    public FullyLayer oLinerLayer;
    
    public DiTSwiGLUFFN mlp;
    
    private int time;
    private int embedDim = 0;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    private int batchSize = 1;
    
    private boolean qkNorm = false;
    private boolean normParams = true;

    public JiTJoinBlockHead(int embedDim, int time, boolean bias, boolean qkNorm, boolean normParams, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.qkNorm = qkNorm;
        this.normParams = normParams;
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

    	this.norm1 = new RMSLayer(1, 1, embedDim, normParams, BNType.fully_bn, network);

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
        
        this.norm2 = new RMSLayer(1, 1, embedDim, normParams, BNType.fully_bn, network);
        this.setoLinerLayer(new FullyLayer(embedDim, embedDim, true, this.network));
        RandomUtils.xavier_uniform(this.oLinerLayer.weight, 1, embedDim, embedDim);
        if(this.oLinerLayer.bias != null) {
        	this.oLinerLayer.bias.clearGPU();
        }
        
        int swiNum = (int) (2.6667 * embedDim);
        this.mlp = new DiTSwiGLUFFN(embedDim, swiNum, embedDim, false, network);
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
            // [batch_size, len_q, n_heads * dim_v]
        	this.output = Tensor.createGPUTensor(this.output, input.number, input.channel, input.height, input.width, true);
        }
        if (this.getqLinerLayer().getOutput() != null) {
            this.getqLinerLayer().getOutput().viewOrg();
            this.getkLinerLayer().getOutput().viewOrg();
            this.getvLinerLayer().getOutput().viewOrg();
            this.getoLinerLayer().getOutput().viewOrg();
        }
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
        // TODO Auto-generated method stub

    }
    
    public void pre_attention(Tensor input) {
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 设置输入
         */
        this.setInput(input);

        norm1.forward(input);

        Tensor x = norm1.getOutput();

        this.getqLinerLayer().forward(x);
        this.getkLinerLayer().forward(x);
        this.getvLinerLayer().forward(x);

    }
    
    public void pre_attention_back(Tensor xDiff, Tensor dq, Tensor dk, Tensor dv) {
    	dq =  dq.view(batchSize * time, 1, 1, width);
    	dk =  dk.view(batchSize * time, 1, 1, width);
    	dv =  dv.view(batchSize * time, 1, 1, width);
    	
    	qLinerLayer.back(dq);
    	kLinerLayer.back(dk);
    	vLinerLayer.back(dv);
    	
    	Tensor dattnInput = qLinerLayer.diff;
    	Tensor_OP().add(dattnInput, kLinerLayer.diff, dattnInput);
    	Tensor_OP().add(dattnInput, vLinerLayer.diff, dattnInput);
    	
    	norm1.back(dattnInput);
    	
    	Tensor_OP().add(norm1.diff, xDiff, norm1.diff);
    	
    	this.diff = norm1.diff;
    }
    
    public void post_attention(Tensor attn) {
    	
    	oLinerLayer.forward(attn);

    	Tensor_OP().add(input, oLinerLayer.getOutput(), oLinerLayer.getOutput());
    	
    	norm2.forward(oLinerLayer.getOutput());

    	mlp.forward(norm2.getOutput());

    	Tensor_OP().add(mlp.getOutput(), oLinerLayer.getOutput(), output);
    	
    }

    public Tensor post_attention_back(Tensor delta) {

    	mlp.back(delta);

    	norm2.back(mlp.diff);
    	Tensor_OP().add(norm2.diff, delta, norm2.diff);
    	
    	oLinerLayer.back(norm2.diff);
    	
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
    	
    	this.norm1.update();
    	
        this.qLinerLayer.update();
        this.kLinerLayer.update();
        this.vLinerLayer.update();
        
        this.norm2.update();
        this.oLinerLayer.update();
        mlp.update();
        
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

    	norm1.saveModel(outputStream);

        getqLinerLayer().saveModel(outputStream);
        getkLinerLayer().saveModel(outputStream);
        getvLinerLayer().saveModel(outputStream);
        this.norm2.saveModel(outputStream);
        this.oLinerLayer.saveModel(outputStream);
        mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {

    	norm1.loadModel(inputStream, 1, 1, width, BNType.fully_bn);

        getqLinerLayer().loadModel(inputStream);
        getkLinerLayer().loadModel(inputStream);
        getvLinerLayer().loadModel(inputStream);
        this.norm2.loadModel(inputStream, 1, 1, width, BNType.fully_bn);
        this.oLinerLayer.loadModel(inputStream);
        mlp.loadModel(inputStream);
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

