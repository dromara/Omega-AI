package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * NerfEmbedder
 *
 * @author Administrator
 */
public class NerfEmbedder extends Layer {
	
    private int in_channels = 0;
    private int hidden_size_input = 1;
    private int max_freqs;
    private int patch_size;
    private boolean bias = false;

    public FullyLayer linear1;
    
    private float[] pos;
    
    private Tensor dct;
    
    private Tensor tmp;
    
    public NerfEmbedder(int in_channels, int hidden_size_input, int max_freqs, int patch_size, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.in_channels = in_channels;
        this.hidden_size_input = hidden_size_input;
        this.max_freqs = max_freqs;
        this.patch_size = patch_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hidden_size_input;
        this.initLayers();
    }

    public static void main(String[] args) {
    	
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(in_channels + max_freqs * max_freqs, hidden_size_input, bias, network);
        RandomUtils.xavier_uniform(this.linear1.weight, 1, in_channels + max_freqs * max_freqs, hidden_size_input);
        if(this.linear1.bias != null) {
        	this.linear1.bias.clearGPU();
        }
        
        pos = RoPEKernel.precompute_freqs_cis_2d(max_freqs * max_freqs * 2, patch_size, patch_size, 10000, 16);
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if(dct == null || dct.number != this.number) {
            dct = new Tensor(number, patch_size * patch_size, 1, max_freqs * max_freqs, MatrixUtils.repeat(pos, number), true);
            tmp = Tensor.createGPUTensor(tmp, number, input.channel, input.height, input.width + dct.width, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
//    	if(diff == null || diff.number != this.number) {
//    		diff = Tensor.createGPUTensor(diff, number, input.channel, input.height, input.width, true);
//    	}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	Tensor_OP().cat_width(input, dct, tmp, input.width, dct.width);
        linear1.forward(tmp);
        this.output = linear1.getOutput();
    }
    
    public void output_eval() {
        // TODO Auto-generated method stub
    	Tensor_OP().cat_width(input, dct, tmp, input.width, dct.width);
        linear1.forward(tmp);
        this.output = linear1.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	this.linear1.back(this.delta);
//    	Tensor_OP().getByWidth_back(diff, linear1.diff, linear1.diff.shape(), 0);
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
//        if(network.RUN_MODEL == RunModel.EVAL) {
//        	this.output_eval();
//        }else {
//        	this.output();
//        }
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
        linear1.update();
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
        linear1.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
    }
}

