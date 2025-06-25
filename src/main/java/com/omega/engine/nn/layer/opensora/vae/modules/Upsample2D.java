package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.UPSampleLayer2;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Downsample2D
 *
 * @author Administrator
 */
public class Upsample2D extends Layer {
	
	private int scale_factor = 2;
	
    private UPSampleLayer2 up;

    public ConvolutionLayer conv;
    
    private int depth;
    
    private Tensor inputT;
    
    public Upsample2D(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oChannel = conv.oChannel;
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
    }

    public void initLayers() {
    	
    	up = new UPSampleLayer2(channel, height, width, scale_factor, network);
       
        conv = new ConvolutionLayer(channel, channel, up.oWidth, up.oHeight, 3, 3, 1, 1, true, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if(inputT == null || inputT.number != this.number) {
        	inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
        	this.output = Tensor.createGPUTensor(output, number, conv.oChannel * depth, conv.oHeight, conv.oWidth, true);
        }
        if(conv.getOutput() != null) {
        	conv.getOutput().viewOrg();
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

    	Tensor_OP().permute(input, inputT,
    			new int[] {number, channel, depth, height, width},
    			new int[] {number, depth, channel, height, width},
    			new int[]{0, 2, 1, 3, 4});

    	up.forward(inputT);

    	conv.forward(up.getOutput());

    	Tensor_OP().permute(conv.getOutput(), output,
    			new int[] {number, depth, conv.oChannel, conv.oHeight, conv.oWidth},
    			new int[] {number, conv.oChannel, depth, conv.oHeight, conv.oWidth},
    			new int[]{0, 2, 1, 3, 4});
    	
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

    	Tensor_OP().permute(delta, conv.getOutput(),
    			new int[] {number, conv.oChannel, depth, conv.oHeight, conv.oWidth},
    			new int[] {number, depth, conv.oChannel, conv.oHeight, conv.oWidth},
    			new int[]{0, 2, 1, 3, 4});
    	
        conv.back(conv.getOutput(), up.getOutput());
       
        up.back(up.getOutput(), inputT);
        
        Tensor_OP().permute(inputT, input,
    			new int[] {number, depth, channel, height, width},
    			new int[] {number, channel, depth, height, width},
    			new int[]{0, 2, 1, 3, 4});

        this.diff = input;
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

