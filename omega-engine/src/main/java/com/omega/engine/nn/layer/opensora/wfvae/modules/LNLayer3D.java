package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.layer.opensora.wfvae.decoder.WFDecoderUp2;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * LNLayer3D
 *
 * @author Administrator
 */
public class LNLayer3D extends Layer {

    public int depth;
    public int oDepth;
    
    private int[] xShape;
    private int[] tShape;
    
    public LNLayer norm;
    
    private Tensor inputT;
    
    public LNLayer3D(int channel,int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
    }
    
    public LNLayer3D(int channel,int depth, int height, int width, Layer preLayer, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oChannel = channel;
        this.oDepth = depth;
        this.oHeight = height;
        this.oWidth = width;
    }

    public void initLayers() {
    	//int groupNum, int channel, int height, int width, BNType bnType, Layer preLayer
    	norm = new LNLayer(depth * height, width, channel, BNType.fully_bn, network);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(inputT == null || inputT.number != this.number) {
        	inputT = Tensor.createGPUTensor(inputT, number, depth * height, width, channel, true);
        	this.output = Tensor.createGPUTensor(output, number, channel * depth, height, width, true);
        	xShape = new int[] {number, channel, depth, height, width};
        	tShape = new int[] {number, depth, height, width, channel};
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if(inputT == null || inputT.number != this.number) {
        	inputT = Tensor.createGPUTensor(inputT, number, depth * height, width, channel, true);
        	this.output = Tensor.createGPUTensor(output, number, channel * depth, height, width, true);
        	xShape = new int[] {number, channel, depth, height, width};
        	tShape = new int[] {number, depth, height, width, channel};
        }
    }
    
    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.number != this.number) {
    		diff = Tensor.createGPUTensor(diff, number, channel * depth, height, width, true);
    	}
    }
    
    public void initBack(Tensor diff) {
    	this.diff = diff;
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }
    
    public static void loadWeight(Map<String, Object> weightMap, LNLayer3D block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        block.norm.gamma = ClipModelUtils.loadData(block.norm.gamma, weightMap, 1, "norm.weight");
        block.norm.beta = ClipModelUtils.loadData(block.norm.beta, weightMap, 1, "norm.bias");
    }
    
    public static void main(String[] args) {
    	int N = 2;
		int C = 16;
		int D = 3;
		int H = 32;
		int W = 32;
    	Tensor input = new Tensor(N, C * D, H, W, true);
    	
        Transformer tf = new Transformer();
        tf.CUDNN = true;
//        float[] data = RandomUtils.order(input.dataLength, 0.1f, 0.1f);
        String inputsPath = "D:\\models\\x2.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(inputsPath);
	    ClipModelUtils.loadData(input, datas2, "x2", 5);
	    
        LNLayer3D norm = new LNLayer3D(C, D, H, W, tf);
        
    	String path = "D:\\models\\LNLayer3D.json";
    	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), norm, true);
        
        for (int i = 0; i < 10; i++) {
        	norm.forward(input);
        	norm.getOutput().showShape();
        	norm.getOutput().showDM();
        	norm.back(input);
        	norm.diff.showDM();
        }
    }
    
    @Override
    public void output() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(input, inputT, xShape, tShape, new int[] {0, 2, 3, 4, 1});
    	norm.forward_llmc(inputT);
    	Tensor_OP().permute(norm.getOutput(), output, tShape, xShape, new int[] {0, 4, 1, 2, 3});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, norm.getOutput(), xShape, tShape, new int[] {0, 2, 3, 4, 1});
    	norm.back(norm.getOutput());
    	Tensor_OP().permute(norm.diff, diff, tShape, xShape, new int[] {0, 4, 1, 2, 3});
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
    	norm.update();
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
    
    public void back(Tensor delta,Tensor diff) {
        // TODO Auto-generated method stub
        initBack(diff);
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
    	norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm.loadModel(inputStream);
    }
}

