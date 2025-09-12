package com.omega.engine.nn.layer.opensora.wfvae.modules;

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
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * Resnet3DBlock
 *
 * @author Administrator
 */
public class WFResnet3DBlock extends Layer {
	
	public int depth;
	public int oDepth;
	
    public LNLayer3D norm1;
    public SiLULayer act1;
    public WFCausalConv3D conv1;
    
    public LNLayer3D norm2;
    public SiLULayer act2;
    public WFCausalConv3D conv2;
    
    public WFCausalConv3D shortcut;
    
    private Tensor tmp_diff;
    private Tensor tmp_norm_diff;
    
    public WFResnet3DBlock(int channel, int oChannel, int depth, int height, int width, Network network) {
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
    	
    	norm1 = new LNLayer3D(channel, depth, height, width, network);
    	act1 = new SiLULayer(norm1);
    	conv1 = new WFCausalConv3D(channel, oChannel, depth, width, height, 3, 1, 1, true, network);
    	conv1.setUpdater(UpdaterFactory.create(this.network));
    	conv1.paramsInit = ParamsInit.silu;
    	
    	norm2 = new LNLayer3D(conv1.oChannel, conv1.oDepth, conv1.oHeight, conv1.oWidth, network);
    	act2 = new SiLULayer(norm2);
    	conv2 = new WFCausalConv3D(oChannel, oChannel, conv1.oDepth, conv1.oWidth, conv1.oHeight, 3, 1, 1, true, network);
    	conv2.setUpdater(UpdaterFactory.create(this.network));
    	conv2.paramsInit = ParamsInit.silu;
    	
    	if(channel != oChannel) {
    		shortcut = new WFCausalConv3D(channel, oChannel, depth, width, height, 1, 1, 0, true, network);
    		shortcut.setUpdater(UpdaterFactory.create(this.network));
    		shortcut.paramsInit = ParamsInit.silu;
    	}
    	
    }

    @Override
    public void init() {
        this.number = this.network.number;
        
        if(this.output == null || this.output.number != this.number) {
        	this.output = Tensor.createGPUTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.number != this.number) {
        	this.diff = Tensor.createGPUTensor(this.diff, number, channel * depth, height, width, true);
        }
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
        
        WFResnet3DBlock block = new WFResnet3DBlock(C, OC, F, H, W, nn);
        
    	String path = "H:\\model\\Resnet3DBlock.json";
    	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), block, true);
        
    	block.forward(input);
    	
    	block.getOutput().showDM();
    	
        float[] data2 = RandomUtils.order(N * C * F * H * W, 0.001f, 0.001f);
        Tensor delta = new Tensor(N, C * F, H, W, data2, true);
        
        block.back(delta);
    	
        block.diff.showDM();
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFResnet3DBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        block.norm1.norm.gamma = ModeLoaderlUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "norm1.weight");
    	block.norm1.norm.beta = ModeLoaderlUtils.loadData(block.norm1.norm.beta, weightMap, 1, "norm1.bias");
    	ModeLoaderlUtils.loadData(block.conv1.weight, weightMap, "conv1.conv.weight", 5);
        ModeLoaderlUtils.loadData(block.conv1.bias, weightMap, "conv1.conv.bias");
        block.norm2.norm.gamma = ModeLoaderlUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "norm2.weight");
    	block.norm2.norm.beta = ModeLoaderlUtils.loadData(block.norm2.norm.beta, weightMap, 1, "norm2.bias");
    	ModeLoaderlUtils.loadData(block.conv2.weight, weightMap, "conv2.conv.weight", 5);
        ModeLoaderlUtils.loadData(block.conv2.bias, weightMap, "conv2.conv.bias");
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
//    	this.tmp_diff = CUDAMemoryManager.getCache("opensora_block_tmp_diff", conv2.input.shape());
//    	delta.showShape("delta---->");
    	conv2.back(delta, conv2.input);
//    	conv2.diff.showDMByOffsetRed(0, 10, "conv2:");
    	act2.back(conv2.diff);
//    	this.tmp_norm_diff = CUDAMemoryManager.getCache("opensora_block_tmp_norm_diff", act2.input.shape());
//    	act2.diff.showDMByOffsetRed(0, 10, "act2:");
    	norm2.back(act2.diff, act2.diff);
    	
//    	this.tmp_diff = CUDAMemoryManager.getCache("opensora_block_tmp_diff", conv1.input.shape());
    	conv1.back(norm2.diff, conv1.input);
    	act1.back(conv1.diff);
    	norm1.back(act1.diff, act1.diff);
    	
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

