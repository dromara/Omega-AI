package com.omega.engine.nn.layer.dit;

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
 * DiTTimeBlock
 * @author Administrator
 */
public class DiTTimeBlock extends Layer {
	
    private int timeDim;
    private int hiddenSize;
    
    private boolean bias = false;
    
    private SiLULayer act;
    public FullyLayer shift_msa_l;
    public FullyLayer scale_msa_l;
    public FullyLayer gate_msa_l;
    public FullyLayer shift_mlp_l;
    public FullyLayer scale_mlp_l;
    public FullyLayer gate_mlp_l;
    

    public DiTTimeBlock(int timeDim, int hiddenSize, boolean bias) {
        this.timeDim = timeDim;
        this.hiddenSize = hiddenSize;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public DiTTimeBlock(int timeDim, int hiddenSize, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.timeDim = timeDim;
        this.hiddenSize = hiddenSize;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.act = new SiLULayer(network);
        this.shift_msa_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.shift_msa_l.bias.clearGPU();
        this.scale_msa_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.scale_msa_l.bias.clearGPU();
        this.gate_msa_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.gate_msa_l.bias.clearGPU();
        this.shift_mlp_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.shift_mlp_l.bias.clearGPU();
        this.scale_mlp_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.scale_mlp_l.bias.clearGPU();
        this.gate_mlp_l = new FullyLayer(timeDim, hiddenSize, bias, network);
        this.gate_mlp_l.bias.clearGPU();
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
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
    	act.forward(input);
    	shift_msa_l.forward(act.getOutput());
    	scale_msa_l.forward(act.getOutput());
    	gate_msa_l.forward(act.getOutput());
    	shift_mlp_l.forward(act.getOutput());
    	scale_mlp_l.forward(act.getOutput());
    	gate_mlp_l.forward(act.getOutput());
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	/**
    	 * this layer is fisrt time layer, so this layer's diff not to need complate.
    	 */
    	shift_msa_l.back(delta);
    	scale_msa_l.back(delta);
    	gate_msa_l.back(delta);
    	shift_mlp_l.back(delta);
    	scale_mlp_l.back(delta);
    	gate_mlp_l.back(delta);
    }
    
    public Tensor getShiftMAS(){
    	return shift_msa_l.getOutput();
    }
    
    public Tensor getScaleMSA(){
    	return scale_msa_l.getOutput();
    }
    
    public Tensor getGateMSA(){
    	return gate_msa_l.getOutput();
    }
    
    public Tensor getShiftMLP(){
    	return shift_mlp_l.getOutput();
    }
    
    public Tensor getScaleMLP(){
    	return scale_mlp_l.getOutput();
    }
    
    public Tensor getGateMLP(){
    	return gate_mlp_l.getOutput();
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
    	shift_msa_l.update();
    	scale_msa_l.update();
    	gate_msa_l.update();
    	shift_mlp_l.update();
    	scale_mlp_l.update();
    	gate_mlp_l.update();
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
    	shift_msa_l.saveModel(outputStream);
    	scale_msa_l.saveModel(outputStream);
    	gate_msa_l.saveModel(outputStream);
    	shift_mlp_l.saveModel(outputStream);
    	scale_mlp_l.saveModel(outputStream);
    	gate_mlp_l.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	shift_msa_l.loadModel(inputStream);
    	scale_msa_l.loadModel(inputStream);
    	gate_msa_l.loadModel(inputStream);
    	shift_mlp_l.loadModel(inputStream);
    	scale_mlp_l.loadModel(inputStream);
    	gate_mlp_l.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	shift_msa_l.accGrad(scale);
    	scale_msa_l.accGrad(scale);
    	gate_msa_l.accGrad(scale);
    	shift_mlp_l.accGrad(scale);
    	scale_mlp_l.accGrad(scale);
    	gate_mlp_l.accGrad(scale);
    }
   
}

