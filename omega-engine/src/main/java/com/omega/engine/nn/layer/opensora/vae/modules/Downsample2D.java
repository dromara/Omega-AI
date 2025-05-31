package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Downsample2D
 *
 * @author Administrator
 */
public class Downsample2D extends Layer {

	public PaddingKernel kernel;

    public ConvolutionLayer conv;

    private Tensor padOutput;
    
    private int[] padding = new int[] {0, 1, 0, 1};

    private int pw;
    private int ph;
    
    private int depth;
    
    private Tensor inputT;
    
    public Downsample2D(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
        this.oChannel = conv.oChannel;
    }

    public void initLayers() {
    	
        kernel = new PaddingKernel(cuda());
        
        pw = padding[0] + width + padding[1];
        ph = padding[2] + height + padding[3];
        
        conv = new ConvolutionLayer(channel, channel, pw, ph, 3, 3, 0, 2, true, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(padOutput == null || padOutput.number != this.number * depth) {
        	inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
        	padOutput = Tensor.createGPUTensor(padOutput, number * depth, channel, ph, pw, true);
        	this.output = Tensor.createGPUTensor(output, number, conv.oChannel * depth, conv.oHeight, conv.oWidth, true);
        }
    }

    @Override
    public void initBack() {
//    	if(this.diff == null || this.diff.number != this.number) {
//    		this.diff = Tensor.createGPUTensor(this.diff, number, channel, height, width, true);
//    	}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	input.view(number, channel, depth, height * width);
    	inputT.view(number, depth, channel, height * width);
    	Tensor_OP().permute(input, inputT, new int[]{0, 2, 1, 3});
    	inputT.viewOrg();
    	kernel.padding2d(inputT, padOutput, padding, 0);
    	
    	conv.forward(padOutput);
    	
    	conv.getOutput().view(number, depth, conv.oChannel, conv.oHeight * conv.oWidth);
    	output.view(number, conv.oChannel, depth, conv.oHeight * conv.oWidth);
    	Tensor_OP().permute(conv.getOutput(), output, new int[]{0, 2, 1, 3});
    	
        output.viewOrg();
        
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	delta.view(number, conv.oChannel, depth, conv.oHeight * conv.oWidth);
    	conv.getOutput().view(number, depth, conv.oChannel, conv.oHeight * conv.oWidth);
    	Tensor_OP().permute(delta, conv.getOutput(), new int[]{0, 2, 1, 3});
    	conv.getOutput().viewOrg();
        conv.back(conv.getOutput(), padOutput);
        
        kernel.padding2dGrad(padOutput, inputT, padding);
        
        inputT.view(number, depth, channel, height * width);
        input.view(number, channel, depth, height * width);
        Tensor_OP().permute(inputT, input, new int[]{0, 2, 1, 3});
        this.diff = input.viewOrg();
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
        conv.update();
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
        conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
    }
}

