package com.omega.engine.nn.layer.dit.mmjit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.org.DiTSwiGLUFFN;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * PlainTextBlock
 *
 * @author Administrator
 */
public class PlainTextBlock extends Layer {
	
    public RMSLayer norm1;
    public RMSLayer norm2;
	
	public AttentionLayer attn;
    
    public DiTSwiGLUFFN mlp;
    
    private int time;
    private int headNum;
    private int embedDim = 0;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;

    private boolean qkNorm = false;

    public PlainTextBlock(int embedDim, int time, int headNum, boolean bias, boolean qkNorm, Network network) {
        this.bias = bias;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.headNum = headNum;
        this.time = time;
        this.embedDim = embedDim;
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
    	norm1 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    	norm2 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    	this.attn = new AttentionLayer(embedDim, headNum, time, bias, qkNorm, network);
        int swiNum = (int) (2.6667 * embedDim);
        this.mlp = new DiTSwiGLUFFN(embedDim, swiNum, embedDim, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
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
    
    public void output(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
    	this.norm1.forward(input);
    	this.attn.forward(norm1.getOutput(), cos, sin);
    	Tensor_OP().add(attn.getOutput(), input, attn.getOutput());
    	this.norm2.forward(attn.getOutput());
    	this.mlp.forward(norm2.getOutput());
    	Tensor_OP().add(mlp.getOutput(), attn.getOutput(), mlp.getOutput());
    	this.output = mlp.getOutput();
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
    
    public void diff(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
    	mlp.back(delta);
    	norm2.back(mlp.diff);
    	Tensor_OP().add(norm2.diff, delta, norm2.diff);
    	attn.back(norm2.diff, cos, sin);
    	norm1.back(attn.diff);
    	this.diff = norm1.diff;
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
    
    public void forward(Tensor input, Tensor cos, Tensor sin) {
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
        this.output(cos, sin);
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
    
    public void back(Tensor delta, Tensor cos, Tensor sin) {
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
    	this.norm1.update();
    	this.norm2.update();
    	this.attn.update();
        this.mlp.update();
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
    	norm2.saveModel(outputStream);
    	attn.saveModel(outputStream);
        this.mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        this.norm1.loadModel(inputStream);
        this.norm2.loadModel(inputStream);
        this.attn.loadModel(inputStream);
        this.mlp.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        norm1.accGrad(scale);
        norm2.accGrad(scale);
        attn.accGrad(scale);
        mlp.accGrad(scale);
    }
}

