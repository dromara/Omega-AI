package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Resnet3DBlock
 *
 * @author Administrator
 */
public class Resnet3DBlock extends Layer {
	
	public int depth;
	public int oDepth;
	
    public GNLayer3D norm1;
    public SiLULayer act1;
    public CausalConv3DPlainAR conv1;
    
    public GNLayer3D norm2;
    public SiLULayer act2;
    public CausalConv3DPlainAR conv2;
    
    public CausalConv3DPlainAR shortcut;
    
    public Resnet3DBlock(int channel, int oChannel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.oChannel = oChannel;
        initLayers();
        this.oDepth = conv2.oDepth;
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    public void initLayers() {
    	
    	norm1 = new GNLayer3D(channel, depth, height, width, 32, network);
    	act1 = new SiLULayer(norm1);
    	conv1 = new CausalConv3DPlainAR(channel, oChannel, depth, width, height, 3, 1, true, network);
    	conv1.setUpdater(UpdaterFactory.create(this.network));
    	conv1.paramsInit = ParamsInit.silu;
    	
    	norm2 = new GNLayer3D(conv1.oChannel, conv1.oDepth, conv1.oHeight, conv1.oWidth, 32, network);
    	act2 = new SiLULayer(norm2);
    	conv2 = new CausalConv3DPlainAR(oChannel, oChannel, conv1.oDepth, conv1.oWidth, conv1.oHeight, 3, 1, true, network);
    	conv2.setUpdater(UpdaterFactory.create(this.network));
    	conv2.paramsInit = ParamsInit.silu;
    	
    	if(channel != oChannel) {
    		shortcut = new CausalConv3DPlainAR(channel, oChannel, depth, width, height, 1, 1, true, network);
    		shortcut.setUpdater(UpdaterFactory.create(this.network));
    		shortcut.paramsInit = ParamsInit.silu;
    	}
    	
    }

    @Override
    public void init() {
        this.number = this.network.number;
        
        if(this.output == null || this.output.getShape()[0] != this.number) {
        	this.output = Tensor.createGPUTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.getShape()[0] != this.number) {
        	this.diff = Tensor.createGPUTensor(this.diff, number, channel * depth, height, width, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub

    	norm1.forward(input);
    	act1.forward(norm1.getOutput());
    	conv1.forward(act1.getOutput());
    	
    	norm2.forward(conv1.getOutput());
    	act2.forward(norm2.getOutput());
    	conv2.forward(act2.getOutput());
    	
    	if(channel != oChannel) {
    		shortcut.forward(input);
    		Tensor_OP().add(conv2.getOutput(), shortcut.getOutput(), this.output);
    	}else {
    		Tensor_OP().add(conv2.getOutput(), input, this.output);
    	}
    	
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	conv2.back(delta);
    	act2.back(conv2.diff);
    	norm2.back(act2.diff);
    	
    	conv1.back(norm2.diff);
    	act1.back(conv1.diff);
    	norm1.back(act1.diff);
    	
    	if(channel != oChannel) {
    		shortcut.back(delta);
    		Tensor_OP().add(norm1.diff, shortcut.diff, norm1.diff);
    	}else {
    		Tensor_OP().add(norm1.diff, delta, norm1.diff);
    	}
    	
    	this.diff = norm1.diff;
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
        if(channel != oChannel) {
        	shortcut.update();
        }
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
        if(channel != oChannel) {
        	shortcut.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv1.loadModel(inputStream);
        norm1.loadModel(inputStream);
        conv2.loadModel(inputStream);
        norm2.loadModel(inputStream);
        if(channel != oChannel) {
        	shortcut.loadModel(inputStream);
        }
    }
}

