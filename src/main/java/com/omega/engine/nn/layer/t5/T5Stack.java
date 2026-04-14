package com.omega.engine.nn.layer.t5;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * T5Stack
 * @author Administrator
 */
public class T5Stack extends Layer {
	
	private int voc_size = 250112;
	private int num_layers = 24;
	private int headNum = 32;
	private int time = 120;
    private int embed_size = 2048;
    private int d_ff = 5120;
    private boolean bias = false;

    public EmbeddingIDLayer embed_tokens;
    public List<T5Block> block;
    public LNLayer final_layer_norm;
    
    public T5Stack(int voc_size, int headNum, int time,int embed_size, int d_ff, int num_layers, boolean bias) {
    	this.voc_size = voc_size;
    	this.num_layers = num_layers;
        this.embed_size = embed_size;
        this.d_ff = d_ff;
        this.headNum = headNum;
        this.time = time;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.initLayers();
    }

    public T5Stack(int voc_size, int headNum, int time,int embed_size, int d_ff, int num_layers, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
    	this.voc_size = voc_size;
    	this.num_layers = num_layers;
        this.embed_size = embed_size;
        this.d_ff = d_ff;
        this.headNum = headNum;
        this.time = time;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.initLayers();
    }

    public static void main(String[] args) {
    	
    }

    public void initLayers() {
    	this.embed_tokens = new EmbeddingIDLayer(voc_size, embed_size, network);
    	this.block = new ArrayList<T5Block>();
    	for(int i = 0;i<num_layers;i++) {
    		boolean has_relative_attention_bias = false;
    		if(i == 0) {
    			has_relative_attention_bias = true;
    		}
    		T5Block t5b = new T5Block(headNum, time, embed_size, d_ff, bias, has_relative_attention_bias, network);
    		block.add(t5b);
    	}
        this.final_layer_norm = new LNLayer(1, 1, embed_size, true, false, BNType.fully_bn, network);
    }
    
    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
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
    
    public void output(Tensor mask) {
    	embed_tokens.forward(input);
    	Tensor bi = embed_tokens.getOutput();
//    	bi.showDM("bi");
    	for(int i = 0;i<num_layers;i++) {
    		T5Block b = block.get(i);
    		if(i > 0) {
    			b.setPosition_bias_masked(block.get(0).getPosition_bias_masked());
    		}
    		b.forward(bi, mask);
    		bi = b.getOutput();
//    		bi.showDM("output:"+i);
    	}
    	final_layer_norm.forward_t5(bi);
    	this.output = final_layer_norm.getOutput();
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
    
    public void forward(Tensor input, Tensor mask) {
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
        this.output(mask);
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
    	embed_tokens.saveModel(outputStream);
    	for(int i = 0;i<num_layers;i++) {
    		block.get(i).saveModel(outputStream);
    	}
        final_layer_norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	embed_tokens.loadModel(inputStream);
     	for(int i = 0;i<num_layers;i++) {
    		block.get(i).loadModel(inputStream);
    	}
    	final_layer_norm.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }
    
}

