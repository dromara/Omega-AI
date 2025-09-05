package com.omega.engine.nn.layer.dit.mmdit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_PoswiseFeedForward Layer
 *
 * @author Administrator
 */
public class MMDiTMLPLayer extends Layer {
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;

    public FullyLayer linear1;
    private GeluLayer active;
    public FullyLayer linear2;
    
    private Tensor tmp1;
    private Tensor tmp2;

    public MMDiTMLPLayer(int embedDim, int nChannel, boolean bias) {
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public MMDiTMLPLayer(int embedDim, int nChannel, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
    }

    public void initLayers() {
        this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);
        this.linear1.weight.setData(RandomUtils.xavierUniform(embedDim * nChannel, embedDim, nChannel, 1.0f));
        if(this.linear1.bias != null) {
        	this.linear1.bias.clearGPU();
        }
        this.active = new GeluLayer(linear1);
        this.linear2 = new FullyLayer(nChannel, embedDim, bias, network);
        this.linear2.weight.setData(RandomUtils.xavierUniform(embedDim * nChannel, nChannel, embedDim, 1.0f));
        if(this.linear2.bias != null) {
        	this.linear2.bias.clearGPU();
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if(network.RUN_MODEL == RunModel.EVAL) {
        	this.tmp1 =  CUDAMemoryManager.getCache("dit_block_mlp_tmp1", number, linear1.oChannel, linear1.oHeight, linear1.oWidth);
        	this.tmp2 =  CUDAMemoryManager.getCache("dit_block_mlp_tmp2", number, linear2.oChannel, linear2.oHeight, linear2.oWidth);
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
        linear1.forward(input);
        active.forward(linear1.getOutput());
        linear2.forward(active.getOutput());
        this.output = linear2.getOutput();
    }
    
    public void output_eval() {
        // TODO Auto-generated method stub
        linear1.forward(input, tmp1);
        active.forward(linear1.getOutput(), tmp1);
        linear2.forward(active.getOutput(), tmp2);
        this.output = linear2.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
//    	delta.showDMByOffsetRed(0, 10, "mlp:delta");
    	this.linear2.back(this.delta);
//    	linear2.diff.showDMByOffsetRed(0, 10, "linear2.diff");
        active.back(this.linear2.diff);
//        active.diff.showDMByOffsetRed(0, 10, "active.diff");
        linear1.back(active.diff);
//        linear1.diff.showDMByOffsetRed(0, 10, "linear1.diff");
        this.diff = this.linear1.diff;
        //		System.out.println("mlp diff:");
        //		diff.showDMByNumber(0);
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
        linear2.update();
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
        linear2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }
}

