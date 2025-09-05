package com.omega.engine.nn.layer.dit.org;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiTSwiGLUFFN
 * @author Administrator
 */
public class DiTSwiGLUFFN extends Layer {
	
    private int inChannel = 0;
    private int hiddenSize;
    private int outChannel;

    private boolean bias = false;
    
    private SiLULayer act;
    public FullyLayer w12;
    public FullyLayer w3;
    
    private Tensor w1;
    private Tensor w2;
    
    private Tensor wt;

    public DiTSwiGLUFFN(int inChannel, int hiddenSize, int outChannel, boolean bias) {
        this.inChannel = inChannel;
        this.hiddenSize = hiddenSize;
        this.outChannel = outChannel;
        this.bias = bias;
        this.channel = 1;
        this.height = 1;
        this.width = inChannel;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = outChannel;
        this.initLayers();
    }

    public DiTSwiGLUFFN(int inChannel, int hiddenSize, int outChannel, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.inChannel = inChannel;
        this.hiddenSize = hiddenSize;
        this.outChannel = outChannel;
        this.bias = bias;
        this.channel = 1;
        this.height = 1;
        this.width = inChannel;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = outChannel;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.act = new SiLULayer(network);

        this.w12 = new FullyLayer(inChannel, hiddenSize * 2, bias, network);
        
        this.w3 = new FullyLayer(hiddenSize, outChannel, bias, network);
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if(w1 == null || w1.number != number) {
        	w1 = Tensor.createGPUTensor(w1, number, 1, 1, hiddenSize, true);
        	w2 = Tensor.createGPUTensor(w2, number, 1, 1, hiddenSize, true);
        	wt = Tensor.createGPUTensor(wt, number, 1, 1, hiddenSize, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(w1 == null || w1.number != number) {
        	w1 = Tensor.createGPUTensor(w1, number, 1, 1, hiddenSize, true);
        	w2 = Tensor.createGPUTensor(w2, number, 1, 1, hiddenSize, true);
        	wt = Tensor.createGPUTensor(wt, number, 1, 1, hiddenSize, true);
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
    	w12.forward(input);
    	
    	int[] shape = new int[] {number, 2, 1, hiddenSize};
    	Tensor_OP().getByChannel(w12.getOutput(), w1, shape, 0);
    	Tensor_OP().getByChannel(w12.getOutput(), w2, shape, 1);
    	
    	act.forward(w1);
    	
    	Tensor_OP().mul(act.getOutput(), w2, wt);
    	
    	w3.forward(wt);
    	
    	this.output =  w3.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	w3.back(delta);
    	
    	//wt = actDelta
    	Tensor_OP().mul(w3.diff, w2, wt); 
    	
    	act.back(wt);
    	
    	int[] shape = new int[] {number, 2, 1, hiddenSize};
    	Tensor_OP().setByChannel(w12.getOutput(), act.diff, shape, 0);
    	
    	//wt = w2Delta
    	Tensor_OP().mul(w3.diff, act.getOutput(), wt); 
    	Tensor_OP().setByChannel(w12.getOutput(), wt, shape, 1);
    	
    	w12.back(w12.getOutput());
    	
    	this.diff = w12.diff;
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
        this.init();
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
    	w12.update();
    	w3.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mlp;
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
    	w12.saveModel(outputStream);
    	w3.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	w12.loadModel(inputStream);
    	w3.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	w12.accGrad(scale);
    	w3.accGrad(scale);
    }
    
    public static void main(String[] args) {
    	
    }
    
}

