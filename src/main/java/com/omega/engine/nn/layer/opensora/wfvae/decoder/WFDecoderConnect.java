package com.omega.engine.nn.layer.opensora.wfvae.decoder;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFConv2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

/**
 * WFDecoderConnect1
 *
 * @author Administrator
 */
public class WFDecoderConnect extends Layer {
	
	public int depth;
	public int oDepth;
	
	private int connect_res_layer_num = 1;
	
    public List<WFResnet3DBlock> resBlocks;

    public WFConv2D conv;
    
    public WFDecoderConnect(int channel, int oChannel, int depth, int height, int width, int connect_res_layer_num, Network network) {
        this.network = network;
        this.connect_res_layer_num = connect_res_layer_num;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.oChannel = oChannel;
        initLayers();
        this.oDepth = conv.oDepth;
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
    }

    public void initLayers() {
    	
    	resBlocks = new ArrayList<WFResnet3DBlock>();
    	
    	int id = depth;
    	int ih = height;
    	int iw = width;
    	
    	for(int i = 0;i<connect_res_layer_num;i++) {
    		WFResnet3DBlock block = new WFResnet3DBlock(channel, channel, id, ih, iw, network);
    		resBlocks.add(block);
    		id = block.oDepth;
    		ih = block.oHeight;
    		iw = block.oWidth;
    	}
    	
    	conv = new WFConv2D(channel, oChannel, id, ih, iw, 3, 1, 1, network);
    	
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
    
    public static void loadWeight(Map<String, Object> weightMap, WFDecoderConnect block, boolean showLayers) {
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
    	
    	conv.forward(x);
    	
    	this.output = conv.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

    	conv.back(delta);
    	
    	Tensor d = conv.diff;
    	
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
        for(int i = 0;i<resBlocks.size();i++) {
        	resBlocks.get(i).saveModel(outputStream);
    	}
        conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        for(int i = 0;i<resBlocks.size();i++) {
        	resBlocks.get(i).loadModel(inputStream);
    	}
        conv.loadModel(inputStream);
    }
}

