package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * Res3DBlockUpsample
 *
 * @author Administrator
 */
public class Res3DBlockUpsample extends Layer {
	
	public int depth;
	public int oDepth;
	
	public CausalConv3DPlainAR conv1;
	public GNLayer3D norm1;
	public SiLULayer act1;
    
	public CausalConv3DPlainAR conv2;
	public GNLayer3D norm2;
	public SiLULayer act2;
    
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
    	norm2.diff.showDM("norm2");
    	conv2.back(norm2.diff);
    	conv2.diff.showDM("conv2");
    	act1.back(conv2.diff);
    	
    	norm1.back(act1.diff);
    	conv1.back(norm1.diff);
    	conv1.diff.showDM("conv1");
    	act2.diff.showDM("act2");
    	Tensor_OP().add(conv1.diff, act2.diff, conv1.diff);
    	conv1.diff.showDM("x");
    	this.diff = conv1.diff;
    }
    
    public static void main(String[] args) {
        int N = 2;
        int C = 32;
        int F = 17;
        int H = 32;
        int W = 32;
        
        float[] data = RandomUtils.order(N * C * F * H * W, 0.01f, 0.01f);
        Tensor input = new Tensor(N, C * F, H, W, data, true);
        Transformer nn = new Transformer();
        nn.CUDNN = true;
        nn.number = N;
        
        Res3DBlockUpsample block = new Res3DBlockUpsample(C, F, H, W, nn);
        
    	String path = "H:\\model\\Res3DBlockUpsample.json";
    	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), block, true);
        
    	block.forward(input);
    	
    	block.getOutput().showDM("output");
    	
        float[] data2 = RandomUtils.order(N * C * F * H * W, 0.001f, 0.001f);
        Tensor delta = new Tensor(N, C * F, H, W, data2, true);
        
        block.back(delta);
    	
        block.diff.showDM();
    }
    
    public static void loadWeight(Map<String, Object> weightMap, Res3DBlockUpsample block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ClipModelUtils.loadData(block.conv1.weight, weightMap, "conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.conv1.bias, weightMap, "conv1.conv.bias");
        block.norm1.norm.gamma = ClipModelUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "norm1.weight");
    	block.norm1.norm.beta = ClipModelUtils.loadData(block.norm1.norm.beta, weightMap, 1, "norm1.bias");
    	ClipModelUtils.loadData(block.conv2.weight, weightMap, "conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.conv2.bias, weightMap, "conv2.conv.bias");
        block.norm2.norm.gamma = ClipModelUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "norm2.weight");
    	block.norm2.norm.beta = ClipModelUtils.loadData(block.norm2.norm.beta, weightMap, 1, "norm2.bias");

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

