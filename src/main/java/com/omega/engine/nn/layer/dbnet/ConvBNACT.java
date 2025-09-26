package com.omega.engine.nn.layer.dbnet;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SwishLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * ConvBNACT conv + bn + act
 *
 * @author Administrator
 */
public class ConvBNACT extends Layer {

    private boolean bias = false;
    
    private int kSize;
    
    public ConvolutionLayer conv1;
    public BNLayer bn;
    private ActiveFunctionLayer act1;
    
    private String actFn;
    
    public ConvBNACT(int in_channels, int out_channels, int height, int width, int kSize, boolean bias, String actFn, Network network) {
        this.channel = in_channels;
        this.height = height;
        this.width = width;
        this.kSize = kSize;
        this.bias = bias;
        this.oChannel = out_channels;
        this.actFn = actFn;
        this.initLayers();
        this.oHeight = conv1.oHeight;
        this.oWidth = conv1.oWidth;
    }

    public void initLayers() {
        this.conv1 = new ConvolutionLayer(channel, oChannel, width, height, kSize, kSize, 0, 1, bias, network);
    	this.bn = new BNLayer(network, BNType.conv_bn);
    	if(actFn.equals("relu")) {
    		this.act1 = new ReluLayer(network);
    	}else {
    		this.act1 = new SwishLayer(network);
    	}
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
    	this.number = input.number;
    	if(output == null || output.number != number) {
    		output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
    	}
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub

    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	conv1.forward(input);
    	bn.forward(conv1.getOutput());
    	act1.forward(bn.getOutput());
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	act1.back(delta);
    	bn.back(act1.diff);
    	conv1.back(bn.diff);
    	this.diff = conv1.diff;
    }
    
    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta();
        /**
         * 计算梯度
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 计算输出
         */
        this.output();
    }
    
    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	conv1.update();
    	bn.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mlp;
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
    	conv1.saveModel(outputStream);
    	bn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv1.loadModel(inputStream);
    	bn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	conv1.accGrad(scale);
    	bn.accGrad(scale);
    }
    
    public static void main(String[] args) {
    	
//    	String inputPath = "H:\\model\\dit_final.json";
//    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//    	
//        int batchSize = 2;
//        int patch_size = 2;
//        int time = 64;
//        int embedDim = 16;
//        int outChannel = 3;
//        
//        Transformer tf = new Transformer();
//        tf.number = batchSize * time;
//        tf.time = time;
//        
//        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
//        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
//        
//        float[] cData = RandomUtils.order(batchSize * embedDim, 0.1f, 0.1f); 
//        Tensor cond = new Tensor(batchSize , 1, 1, embedDim, cData, true);
//        
//        int ow = patch_size * patch_size * outChannel;
//        
//        float[] delta_data = RandomUtils.order(batchSize * time * ow, 0.01f, 0.01f);
//        Tensor delta = new Tensor(batchSize * time, 1, 1, ow, delta_data, true);
//        
//        Tensor dcond = new Tensor(batchSize, 1, 1, embedDim, true);
//
//        SEBlock finalLayer = new SEBlock(patch_size, embedDim, outChannel, time, true, true, tf);
//        
//        loadWeight(datas, finalLayer, true);
//        
//        for (int i = 0; i < 10; i++) {
//            //			input.showDM();
//        	dcond.clearGPU();
//        	finalLayer.forward(input, cond);
//        	finalLayer.getOutput().showShape();
//        	finalLayer.getOutput().showDM();
//        	finalLayer.back(delta, dcond);
//////            //			delta.showDM();
//        	finalLayer.diff.showDM("dx");
//        	dcond.showDM("dcond");
//            //			delta.copyData(tmp);
//        }
    }
}

