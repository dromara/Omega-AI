package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Res3DBlockUpsample
 *
 * @author Administrator
 */
public class Res3DBlockUpsample extends Layer {
	
	public int depth;
	public int oDepth;
	
    private CausalConv3DPlainAR conv1;
    private GNLayer3D norm1;
    private SiLULayer act1;
    
    private CausalConv3DPlainAR conv2;
    private GNLayer3D norm2;
    private SiLULayer act2;
    
    private Tensor resOut;
    
    public Res3DBlockUpsample(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oChannel = conv2.oChannel;
        this.oDepth = conv2.oDepth;
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    public void initLayers() {
    	
    	conv1 = new CausalConv3DPlainAR(channel, channel, depth, width, height, 3, 1, true, network);
    	conv1.setUpdater(UpdaterFactory.create(this.network));
    	conv1.paramsInit = ParamsInit.silu;
    	
    	norm1 = new GNLayer3D(conv1.oChannel, conv1.oDepth, conv1.oHeight, conv1.oWidth, 32, conv1, network);
    	
    	act1 = new SiLULayer(norm1);
       
    	conv2 = new CausalConv3DPlainAR(channel, channel, conv1.oDepth, conv1.oWidth, conv1.oHeight, 3, 1, true, network);
    	conv2.setUpdater(UpdaterFactory.create(this.network));
    	conv2.paramsInit = ParamsInit.silu;
    	
    	norm2 = new GNLayer3D(conv2.oChannel, conv2.oDepth, conv2.oHeight, conv2.oWidth, 32, conv2, network);
    	
    	act2 = new SiLULayer(norm2);
    	
    }

    @Override
    public void init() {
        this.number = this.network.number;
        
        if(this.resOut == null || this.resOut.number != this.number) {
        	this.resOut = Tensor.createGPUTensor(this.resOut, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
    	
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	conv1.forward(input);
    	norm1.forward(conv1.getOutput());
    	act1.forward(norm1.getOutput());
    	
    	conv2.forward(act1.getOutput());
    	norm2.forward(conv2.getOutput());
    	Tensor_OP().add(norm2.getOutput(), input, this.resOut);
    	act2.forward(resOut);

        this.output = act2.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	act2.back(delta);
    	norm2.back(act2.diff);
    	conv2.back(norm2.diff);
        
    	act1.back(conv2.diff);
    	norm1.back(act1.diff);
    	conv1.back(norm1.diff);
    	Tensor_OP().add(conv1.diff, act2.diff, conv1.diff);
    	
    	this.diff = conv1.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        conv1.update();
        norm1.update();
        conv2.update();
        norm2.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.block;
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
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
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        conv1.saveModel(outputStream);
        norm1.saveModel(outputStream);
        conv2.saveModel(outputStream);
        norm2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv1.loadModel(inputStream);
        norm1.loadModel(inputStream);
        conv2.loadModel(inputStream);
        norm2.loadModel(inputStream);
    }
}

