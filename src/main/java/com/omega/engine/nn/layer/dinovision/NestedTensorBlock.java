package com.omega.engine.nn.layer.dinovision;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * NestedTensorBlock
 *
 * @author Administrator
 */
public class NestedTensorBlock extends Layer {
	
    private boolean bias = false;

    public LNLayer norm1;
    public MemEffAttention attn;
    public LayerScale scale1;
    public LNLayer norm2;
    public DiTMLPLayer mlp;
    public LayerScale scale2;
    
    private int embedDim = 0;
    private int time = 0;
    private int headNum = 0;
    private int mlpHiddenDim;

    public NestedTensorBlock(int embedDim, int time, int mlpHiddenDim, int headNum, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.time = time;
        this.headNum = headNum;
        this.mlpHiddenDim = mlpHiddenDim;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public void initLayers() {
        this.norm1 = new LNLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        this.attn = new MemEffAttention(embedDim, headNum, time, bias, network);
        this.scale1 = new LayerScale(embedDim, network);
        
        this.norm2 = new LNLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        this.mlp = new DiTMLPLayer(embedDim, mlpHiddenDim, bias, network);
        this.scale2 = new LayerScale(embedDim, network);

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
    
    public void output_eval() {
        // TODO Auto-generated method stub
        norm1.forward_llmc(input);
        attn.forward(norm1.getOutput());
        scale1.forward(attn.getOutput());
        Tensor_OP().add(input, scale1.getOutput(), scale1.getOutput());
        
        norm2.forward_llmc(scale1.getOutput());
        mlp.forward(norm2.getOutput());
        scale2.forward(mlp.getOutput());
        Tensor_OP().add(scale1.getOutput(), scale2.getOutput(), scale2.getOutput());
        this.output = scale2.getOutput();
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
        this.output_eval();
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
    	norm1.saveModel(outputStream);
    	attn.saveModel(outputStream);
    	scale1.saveModel(outputStream);
    	norm2.saveModel(outputStream);
    	mlp.saveModel(outputStream);
    	scale2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm1.loadModel(inputStream);
    	attn.loadModel(inputStream);
    	scale1.loadModel(inputStream);
    	norm2.loadModel(inputStream);
    	mlp.loadModel(inputStream);
    	scale2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }
}

