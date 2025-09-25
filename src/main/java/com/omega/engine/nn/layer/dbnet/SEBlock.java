package com.omega.engine.nn.layer.dbnet;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.AdaptiveAvgPool2DLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * SEBlock
 *
 * @author Administrator
 */
public class SEBlock extends Layer {

    private boolean bias = false;
    
    private int num_mid_filter;
    
    public AdaptiveAvgPool2DLayer pool;

    public ConvolutionLayer conv1;
    private ReluLayer act1;
    public ConvolutionLayer conv2;
    public SigmodLayer act2;


    public SEBlock(int in_channels, int out_channels,int ratio, int height, int width, boolean bias, Network network) {
        this.channel = in_channels;
        this.height = height;
        this.width = height;
        this.bias = bias;
        this.oChannel = out_channels;
        this.num_mid_filter = out_channels / ratio;
        this.initLayers();
        this.oHeight = act2.oHeight;
        this.oWidth = act2.oWidth;
    }

    public void initLayers() {
    	this.pool = new AdaptiveAvgPool2DLayer(channel, height, width, 1, 1, network);
        this.conv1 = new ConvolutionLayer(channel, num_mid_filter, pool.oWidth, pool.oHeight, 1, 1, 0, 1, bias, network);
        this.act1 = new ReluLayer(network);
        this.conv2 = new ConvolutionLayer(num_mid_filter, oChannel, act1.oWidth, act1.oHeight, 1, 1, 0, 1, bias, network);
        this.act2 = new SigmodLayer(network);
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
    	pool.forward(input);
    	conv1.forward(pool.getOutput());
    	act1.forward(conv1.getOutput());
    	conv2.forward(act1.getOutput());
    	act2.forward(conv2.getOutput());
    	Tensor_OP().mulAxis(input, act2.getOutput(), output, act2.getOutput().number * act2.getOutput().channel);
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().mulAxisBack(diff, input, act2.input);
    	act2.back(act2.input);
    	conv2.back(act2.diff);
    	act1.back(conv2.diff);
    	conv1.back(act1.diff);
    	pool.back(conv1.diff);
    	Tensor_OP().mulAxis(diff, act2.getOutput(), output, act2.getOutput().number * act2.getOutput().channel);
    	Tensor_OP().add(output, pool.diff, pool.diff);
    	this.diff = pool.diff;
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
    	conv2.update();
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
    	conv2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv1.loadModel(inputStream);
    	conv2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	conv1.accGrad(scale);
    	conv1.accGrad(scale);
    }
//    
//    public static void loadWeight(Map<String, Object> weightMap, SEBlock block, boolean showLayers) {
//        if (showLayers) {
//            for (String key : weightMap.keySet()) {
//                System.out.println(key);
//            }
//        }
//        
//        block.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalNorm.gamma, weightMap, 1, "norm_final.weight");
//        
//        ModeLoaderlUtils.loadData(block.finalLinear.weight, weightMap, "linear.weight");
//        ModeLoaderlUtils.loadData(block.finalLinear.bias, weightMap, "linear.bias");
//        
//        ModeLoaderlUtils.loadData(block.m_linear1.weight, weightMap, "adaLN_modulation1.weight");
//        ModeLoaderlUtils.loadData(block.m_linear1.bias, weightMap, "adaLN_modulation1.bias");
//        
//        ModeLoaderlUtils.loadData(block.m_linear2.weight, weightMap, "adaLN_modulation2.weight");
//        ModeLoaderlUtils.loadData(block.m_linear2.bias, weightMap, "adaLN_modulation2.bias");
//    }
    
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

