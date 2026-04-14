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
public class LTXVideoDownBlock3D extends Layer {

	public int depth;
	public int oDepth;

	public boolean spatio_temporal_scale;
    
	private boolean is_causal;
	
    private int num_layers;
    
    public List<LTXVideoResnetBlock3d> resnets;
    public LTXVideoCausalConv3d downsampler;
    public LTXVideoResnetBlock3d conv_out;

    public boolean shortcut = false;
    
    public LTXVideoDownBlock3D(int channel, int oChannel, int depth, int height, int width, int num_layers, boolean is_causal, boolean spatio_temporal_scale, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.num_layers = num_layers;
        this.spatio_temporal_scale = spatio_temporal_scale;
        this.is_causal = is_causal;
        if (channel != oChannel) {
            shortcut = true;
        }
        initLayers();
    }

    public void initLayers() {
    	
    	resnets = new ArrayList<LTXVideoResnetBlock3d>();
    	
    	int inDepth = depth;
    	int inHeight = height;
     	int inWidth = width;
    	for(int i = 0;i<num_layers;i++) {
    		LTXVideoResnetBlock3d block = new LTXVideoResnetBlock3d(channel, channel, inDepth, inHeight, inWidth, is_causal, network);
    		resnets.add(block);
    		inDepth = block.oDepth;
    		inHeight = block.oHeight;
    		inWidth = block.oWidth;
    	}
    	
    	if(spatio_temporal_scale) {
    		downsampler = new LTXVideoCausalConv3d(channel, channel, inDepth, inWidth, inHeight, 3, 3, 3, 2, true, is_causal, network);
    		inDepth = downsampler.oDepth;
    		inHeight = downsampler.oHeight;
    		inWidth = downsampler.oWidth;
    	}
    	
        if(shortcut) {
        	conv_out = new LTXVideoResnetBlock3d(channel, oChannel, inDepth, inHeight, inWidth, is_causal, network);
        	inDepth = conv_out.oDepth;
    		inHeight = conv_out.oHeight;
    		inWidth = conv_out.oWidth;
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
    	for(int i = 0;i<num_layers;i++) {
    		LTXVideoResnetBlock3d block = resnets.get(i);
    		block.forward(x);
    		x = block.getOutput();
    	}
//    	System.err.println("----1");
    	if(spatio_temporal_scale) {
//        	System.err.println("----2");
    		downsampler.forward(x);
    		x = downsampler.getOutput();
//    		x.showShape("downsampler");
//    		x.showDMByOffsetRed((3 * 9 + 2) * x.height * x.width, x.height * x.width, "downsampler");
    	}
    	
    	if(shortcut) {
//        	System.err.println("----3");
    		conv_out.forward(x);
    		x = conv_out.getOutput();
//    		x.showDMByOffsetRed((3 * 9 + 2) * x.height * x.width, x.height * x.width, "conv_out");
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
        for (int i = 0; i < resnets.size(); i++) {
        	LTXVideoResnetBlock3d l = resnets.get(i);
            l.saveModel(outputStream);
        }
        if(spatio_temporal_scale) {
        	downsampler.saveModel(outputStream);
        }
    	if(shortcut) {
    		conv_out.saveModel(outputStream);
    	}
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        for (int i = 0; i < resnets.size(); i++) {
        	LTXVideoResnetBlock3d l = resnets.get(i);
            l.loadModel(inputStream);
        }
        if(spatio_temporal_scale) {
        	downsampler.loadModel(inputStream);
        }
    	if(shortcut) {
    		conv_out.loadModel(inputStream);
    	}
    }
}

