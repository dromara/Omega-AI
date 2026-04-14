package com.omega.engine.nn.layer.videovae.block;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * LTXVideoDownBlock3D
 *
 * @author Administrator
 */
public class LTXVideoUpBlock3d extends Layer {

	private int upscale_factor;
	
	public int depth;
	public int oDepth;

	public boolean spatio_temporal_scale;
    
	private boolean is_causal;
	
    private int num_layers;
    
    public LTXVideoResnetBlock3d conv_in;
    public LTXVideoUpsampler3d upsampler;
    public List<LTXVideoResnetBlock3d> resnets;

    public boolean shortcut = false;
    
    public LTXVideoUpBlock3d(int channel, int oChannel, int depth, int height, int width, int num_layers, int upscale_factor, boolean is_causal, boolean spatio_temporal_scale, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.num_layers = num_layers;
        this.upscale_factor = upscale_factor;
        this.spatio_temporal_scale = spatio_temporal_scale;
        this.is_causal = is_causal;
        if (channel != oChannel) {
            shortcut = true;
        }
        initLayers();
    }

    public void initLayers() {

    	int inDepth = depth;
    	int inHeight = height;
     	int inWidth = width;
     	int inChannel = channel;
        if(shortcut) {
        	conv_in = new LTXVideoResnetBlock3d(channel, oChannel, inDepth, inHeight, inWidth, is_causal, network);
        	inDepth = conv_in.oDepth;
        	inChannel = oChannel;
    		inHeight = conv_in.oHeight;
    		inWidth = conv_in.oWidth;
        }

    	if(spatio_temporal_scale) {
    		upsampler = new LTXVideoUpsampler3d(inChannel * upscale_factor, inDepth, inHeight, inWidth, new int[] {2, 2, 2}, upscale_factor, false, is_causal, network);
    		inDepth = upsampler.oDepth;
    		inChannel = upsampler.oChannel;
    		inHeight = upsampler.oHeight;
    		inWidth = upsampler.oWidth;
    	}
    	
    	resnets = new ArrayList<LTXVideoResnetBlock3d>();
    	
    	for(int i = 0;i<num_layers;i++) {
    		LTXVideoResnetBlock3d block = new LTXVideoResnetBlock3d(inChannel, inChannel, inDepth, inHeight, inWidth, is_causal, network);
    		resnets.add(block);
    		inDepth = block.oDepth;
    		inHeight = block.oHeight;
    		inWidth = block.oWidth;
    	}
    	
        this.oDepth = inDepth;
		this.oHeight = inHeight;
		this.oWidth = inWidth;
    }

    @Override
    public void init() {
        this.number = this.network.number;
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
    	Tensor x = input;
    	
    	if(shortcut) {
    		conv_in.forward(x);
    		x = conv_in.getOutput();
        	x.showDMByOffsetRed((3 * conv_in.oDepth + 2) * x.height * x.width, x.height * x.width, "shortcut");
    	}
    	System.err.println(spatio_temporal_scale);
    	if(spatio_temporal_scale) {
    		upsampler.forward(x);
    		x = upsampler.getOutput();
    		x.showDMByOffsetRed((3 * upsampler.oDepth + 2) * x.height * x.width, x.height * x.width, "upsampler");
    	}
    	
    	for(int i = 0;i<num_layers;i++) {
    		LTXVideoResnetBlock3d block = resnets.get(i);
    		block.forward(x);
    		x = block.getOutput();
    		x.showShape("x");
    		x.showDMByOffsetRed((3 * block.oDepth + 2) * x.height * x.width, x.height * x.width, "resnet");
    	}

    	this.output = x;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
       
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
    	if(shortcut) {
    		conv_in.saveModel(outputStream);
    	}
    	if(spatio_temporal_scale) {
    		upsampler.saveModel(outputStream);
    	}
    	for(int i = 0;i<resnets.size();i++) {
    		LTXVideoResnetBlock3d block = resnets.get(i);
    		block.saveModel(outputStream);
    	}
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(shortcut) {
    		conv_in.loadModel(inputStream);
    	}
    	if(spatio_temporal_scale) {
    		upsampler.loadModel(inputStream);
    	}
    	for(int i = 0;i<resnets.size();i++) {
    		LTXVideoResnetBlock3d block = resnets.get(i);
    		block.loadModel(inputStream);
    	}
    }
}

