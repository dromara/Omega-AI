package com.omega.engine.nn.layer.opensora.wfvae.decoder;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.opensora.vae.modules.Upsample2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

/**
 * WFDecoderUp1
 *
 * @author Administrator
 */
public class WFDecoderUp1 extends Layer {
	
	public int depth;
	public int oDepth;
	
	private int num_resblocks;
	
    public List<WFResnet3DBlock> resBlocks;

    public Upsample2D up2d;

    public WFResnet3DBlock block;
    
    public WFDecoderUp1(int channel, int oChannel, int depth, int height, int width, int num_resblocks, Network network) {
        this.network = network;
        this.num_resblocks = num_resblocks;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.oChannel = oChannel;
        initLayers();
        this.oDepth = block.oDepth;
        this.oHeight = block.oHeight;
        this.oWidth = block.oWidth;
    }

    public void initLayers() {
    	
    	resBlocks = new ArrayList<WFResnet3DBlock>();
    	
    	int id = depth;
    	int ih = height;
    	int iw = width;
    	
    	for(int i = 0;i<num_resblocks;i++) {
    		WFResnet3DBlock block = new WFResnet3DBlock(i == 0 ? channel : oChannel, oChannel, id, ih, iw, network);
    		resBlocks.add(block);
    		id = block.oDepth;
    		ih = block.oHeight;
    		iw = block.oWidth;
    	}
    	
    	up2d = new Upsample2D(oChannel, id, ih, iw, network);
    	
    	block = new WFResnet3DBlock(oChannel, oChannel, up2d.oDepth, up2d.oHeight, up2d.oWidth, network);
    	
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

    public static void main(String[] args) {
        int N = 2;
        int C = 32;
        int F = 17;
        int H = 32;
        int W = 32;
        
        int OC = 32;
        
        float[] data = RandomUtils.order(N * C * F * H * W, 0.01f, 0.01f);
        Tensor input = new Tensor(N, C * F, H, W, data, true);
        Transformer nn = new Transformer();
        nn.CUDNN = true;
        nn.number = N;
        
//        WFEncoderDown1 block = new WFEncoderDown1(C, OC, F, H, W, nn);
//        
//    	String path = "H:\\model\\Resnet3DBlock.json";
//    	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), block, true);
//        
//    	block.forward(input);
//    	
//    	block.getOutput().showDM();
//    	
//        float[] data2 = RandomUtils.order(N * C * F * H * W, 0.001f, 0.001f);
//        Tensor delta = new Tensor(N, C * F, H, W, data2, true);
//        
//        block.back(delta);
//    	
//        block.diff.showDM();
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFDecoderUp1 block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
//        block.norm1.norm.gamma = ClipModelUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "norm1.weight");
//    	block.norm1.norm.beta = ClipModelUtils.loadData(block.norm1.norm.beta, weightMap, 1, "norm1.bias");
//    	ClipModelUtils.loadData(block.conv1.weight, weightMap, "conv1.conv.weight", 5);
//        ClipModelUtils.loadData(block.conv1.bias, weightMap, "conv1.conv.bias");
//        block.norm2.norm.gamma = ClipModelUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "norm2.weight");
//    	block.norm2.norm.beta = ClipModelUtils.loadData(block.norm2.norm.beta, weightMap, 1, "norm2.bias");
//    	ClipModelUtils.loadData(block.conv2.weight, weightMap, "conv2.conv.weight", 5);
//        ClipModelUtils.loadData(block.conv2.bias, weightMap, "conv2.conv.bias");
    }
    
    @Override
    public void output() {
        // TODO Auto-generated method stub

    	Tensor x = input;
    	
    	for(int i = 0;i<resBlocks.size();i++) {
    		WFResnet3DBlock block = resBlocks.get(i);
    		block.forward(x);
    		x = block.getOutput();
    	}
    	
    	up2d.forward(x);
    	
    	block.forward(up2d.getOutput());
    	
    	this.output = block.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

    	block.back(delta);

    	up2d.back(block.diff);
    	
    	Tensor d = up2d.diff;
    	
    	for(int i = resBlocks.size() - 1;i>=0;i--) {
    		WFResnet3DBlock block = resBlocks.get(i);
    		block.back(d);
    		d = block.diff;
    	}

    	this.diff = d;
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
        for(int i = 0;i<resBlocks.size();i++) {
        	resBlocks.get(i).update();
    	}
        up2d.update();
        block.update();
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
        for(int i = 0;i<resBlocks.size();i++) {
        	resBlocks.get(i).saveModel(outputStream);
    	}
        up2d.saveModel(outputStream);
    	block.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        for(int i = 0;i<resBlocks.size();i++) {
        	resBlocks.get(i).loadModel(inputStream);
    	}
        up2d.loadModel(inputStream);
    	block.loadModel(inputStream);
    }
}

