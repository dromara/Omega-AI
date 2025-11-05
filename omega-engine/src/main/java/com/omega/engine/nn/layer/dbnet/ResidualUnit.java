package com.omega.engine.nn.layer.dbnet;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * ResidualUnit
 *
 * @author Administrator
 */
public class ResidualUnit extends Layer {

    private boolean bias = false;
    
    private int mid_channels;
    
    private int kSize;
    
    private int stride;
    
    private String act;
    
    private boolean use_se;
    
    public ConvBNACT conv0;
    public ConvBNACT conv1;
    public ConvBNACT conv2;
    public SEBlock se;

    private boolean not_add = false;
    
    public ResidualUnit(int in_channels, int mid_channels, int out_channels, int height, int width, int kSize, int stride, String act, boolean use_se, boolean bias, Network network) {
        this.channel = in_channels;
        this.height = height;
        this.width = height;
        this.bias = bias;
        this.stride = stride;
        this.kSize = kSize;
        this.act = act;
        this.use_se = use_se;
        this.oChannel = out_channels;
        this.mid_channels = mid_channels;
        this.initLayers();
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    public void initLayers() {
    	this.conv0 = new ConvBNACT(channel, mid_channels, height, width, 1, 1, 0, bias, act, network);
        this.conv1 = new ConvBNACT(mid_channels, mid_channels, conv0.oHeight, conv0.width, kSize, stride, (kSize - 1)/2, bias, act, network);
        if(use_se) {
        	se = new SEBlock(mid_channels, mid_channels, 4, conv1.oHeight, conv1.oWidth, bias, network);
        }
        this.conv2 = new ConvBNACT(mid_channels, oChannel, conv1.oHeight, conv1.oWidth, 1, 1, 0, bias, act, network);
        if(channel != oChannel || stride != 1) {
        	not_add = true;
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
    	this.number = input.number;

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
    	conv0.forward(input);
    	conv1.forward(conv0.getOutput());
    	Tensor y = conv1.getOutput();
    	if(use_se){
    		se.forward(y);
    		y = se.getOutput();
    	}
    	
    	conv2.forward(y);
    	
    	if(!not_add) {
    		Tensor_OP().add(conv2.getOutput(), input, conv2.getOutput());
    	}
    	
    	this.output = conv2.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    	conv2.back(delta);
    	
    	Tensor d = conv2.diff;
    	
    	if(use_se){
    		se.back(d);
    		d = se.diff;
    	}
    	
    	conv1.back(d);
    	conv0.back(conv1.diff);
    	
    	if(!not_add) {
    		Tensor_OP().add(conv0.diff, delta, conv0.diff);
    	}
    	
    	this.diff = conv0.diff;
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
    	conv0.update();
    	conv1.update();
    	conv2.update();
    	se.update();
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
    	conv0.saveModel(outputStream);
    	conv1.saveModel(outputStream);
    	conv2.saveModel(outputStream);
    	se.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv0.loadModel(inputStream);
    	conv1.loadModel(inputStream);
    	conv2.loadModel(inputStream);
    	se.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	conv0.accGrad(scale);
    	conv1.accGrad(scale);
    	conv2.accGrad(scale);
    	se.accGrad(scale);
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

