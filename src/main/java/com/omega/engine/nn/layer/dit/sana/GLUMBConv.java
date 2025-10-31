package com.omega.engine.nn.layer.dit.sana;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * GLUMBConv
 *
 * @author Administrator
 */
public class GLUMBConv extends Layer {
	
    private int batchSize = 1;
	
    public ConvolutionLayer inverted_conv;
    private SiLULayer inverted_act;
    
    public ConvolutionLayer depth_conv;
    
    public ConvolutionLayer point_conv;
    
    private SiLULayer glu_act;
    
    private int hidden_features;
    
    private int h;
    private int w;
    
    private Tensor xt;
    
    private Tensor xl;
    private Tensor gate;
    
    public GLUMBConv(int embedDim, int in_features, int hidden_features, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.hidden_features = hidden_features;
        this.channel = embedDim;
        this.height = 1;
        this.width = in_features;
        this.oChannel = embedDim;
        this.oHeight = 1;
        this.oWidth = in_features;
        initLayers();
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        this.batchSize = number / channel;
        if (xt == null || xt.number != this.batchSize) {
        	xt = Tensor.createGPUTensor(xt, batchSize, width, h, w, true);
        	xl = Tensor.createGPUTensor(xl, batchSize, hidden_features, h, w, true);
        	gate = Tensor.createGPUTensor(gate, batchSize, hidden_features, h, w, true);
        	output = Tensor.createGPUTensor(output, batchSize * oChannel, 1, 1, oWidth, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = number / channel;
        if (xt == null || xt.number != this.batchSize) {
        	xt = Tensor.createGPUTensor(xt, batchSize, width, h, w, true);
        	xl = Tensor.createGPUTensor(xl, batchSize, hidden_features, h, w, true);
        	gate = Tensor.createGPUTensor(gate, batchSize, hidden_features, h, w, true);
        	output = Tensor.createGPUTensor(output, batchSize * oChannel, 1, 1, oWidth, true);
        }
    }

    public void initLayers() {
        
    	h = (int) Math.sqrt(channel);
    	w = (int) Math.sqrt(channel);
    	
    	this.inverted_conv = new ConvolutionLayer(width, hidden_features*2, w, h, 1, 1, 0, 1, true, network);
    	this.inverted_act = new SiLULayer(inverted_conv);
    	this.depth_conv = new ConvolutionLayer(hidden_features*2, hidden_features*2, inverted_conv.oWidth, inverted_conv.oHeight, 3, 3, 1, 1, true, network);
    	this.depth_conv.setGroups(hidden_features*2);
    	this.point_conv = new ConvolutionLayer(hidden_features, width, depth_conv.oWidth, depth_conv.oHeight, 1, 1, 0, 1, false, network);
    	this.glu_act = new SiLULayer(network);
    	
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
    	Tensor x = input.view(batchSize, h, w, width);

    	Tensor_OP().permute(x, xt, new int[] {0, 3, 1, 2});
    	
    	inverted_conv.forward(xt);
    	inverted_act.forward(inverted_conv.getOutput());
    	
    	depth_conv.forward(inverted_act.getOutput());

    	Tensor_OP().getByChannel(depth_conv.getOutput(), xl, depth_conv.getOutput().shape(), 0, hidden_features);
    	Tensor_OP().getByChannel(depth_conv.getOutput(), gate, depth_conv.getOutput().shape(), hidden_features, hidden_features);
    	
    	glu_act.forward(gate);

    	Tensor_OP().mul(xl, glu_act.getOutput(), xl);
    	
    	point_conv.forward(xl);
    	
    	Tensor_OP().permute(point_conv.getOutput(), output, point_conv.getOutput().shape(), x.shape(), new int[] {0, 2, 3, 1});
    	
    	input.view(batchSize * h * w, 1, 1, width);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	int[] shape = new int[] {batchSize, h, w, width};
    	Tensor_OP().permute(delta, point_conv.getOutput(), shape, point_conv.getOutput().shape(), new int[] {0, 3, 1, 2});
    	
    	point_conv.back(point_conv.getOutput());

    	// 重新获取xl
    	Tensor_OP().getByChannel(depth_conv.getOutput(), xl, depth_conv.getOutput().shape(), 0, hidden_features);
    	// dglu
    	Tensor_OP().mul(point_conv.diff, xl, xl);
    	glu_act.back(xl);
    	Tensor_OP().getByChannel_back(depth_conv.getOutput(), glu_act.diff, depth_conv.getOutput().shape(), hidden_features, hidden_features);
    	//dxl
    	Tensor_OP().mul(point_conv.diff, glu_act.getOutput(), xl);
    	Tensor_OP().getByChannel_back(depth_conv.getOutput(), xl, depth_conv.getOutput().shape(), 0, hidden_features);
    	
    	depth_conv.back(depth_conv.getOutput());
    	
    	inverted_act.back(depth_conv.diff);
    	inverted_conv.back(inverted_act.diff);
    	
    	Tensor_OP().permute(inverted_conv.diff, output, inverted_conv.diff.shape(), shape, new int[] {0, 2, 3, 1});
    	
    	this.diff = output;
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
    	inverted_conv.update();
    	depth_conv.update();
    	point_conv.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.clip_vision_embedding;
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
    	inverted_conv.saveModel(outputStream);
    	depth_conv.saveModel(outputStream);
    	point_conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	inverted_conv.loadModel(inputStream);
    	depth_conv.loadModel(inputStream);
    	point_conv.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	inverted_conv.accGrad(scale);
    	depth_conv.accGrad(scale);
    	point_conv.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, GLUMBConv block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(block.inverted_conv.weight, weightMap, "inverted_conv.conv.weight", 4);
        ModeLoaderlUtils.loadData(block.inverted_conv.bias, weightMap, "inverted_conv.conv.bias");
        ModeLoaderlUtils.loadData(block.depth_conv.weight, weightMap, "depth_conv.conv.weight", 4);
        ModeLoaderlUtils.loadData(block.depth_conv.bias, weightMap, "depth_conv.conv.bias");
        ModeLoaderlUtils.loadData(block.point_conv.weight, weightMap, "point_conv.conv.weight", 4);

    }
    
    public static void main(String[] args) {
    	String inputPath = "D:\\models\\mlp_w.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	
        int batchSize = 2;
        int time = 16;
        int embedDim = 32;

        float mlpRate = 2.5f;
        int hidSize = (int) (embedDim * mlpRate);
        Transformer tf = new Transformer();
        tf.CUDNN = true;
        tf.number = batchSize * time;
        tf.time = time;
        
        GLUMBConv lite = new GLUMBConv(time, embedDim, hidSize, tf);
        
        loadWeight(datas, lite, true);
        
	    String xPath = "D:\\models\\mlp_x.json";
	    Map<String, Object> xDatas = LagJsonReader.readJsonFileSmallWeight(xPath);
	    Tensor input = new Tensor(batchSize, time, 1, embedDim, true);
	    ModeLoaderlUtils.loadData(input, xDatas, "x", 3);
	    input.view(batchSize * time, 1, 1, embedDim);
	    
	    String dxPath = "D:\\models\\mlp_dx.json";
	    Map<String, Object> dxDatas = LagJsonReader.readJsonFileSmallWeight(dxPath);
	    Tensor delta = new Tensor(batchSize, time, 1, embedDim, true);
	    ModeLoaderlUtils.loadData(delta, dxDatas, "dx", 3);
	    delta.view(batchSize * time, 1, 1, embedDim);
	    
        for (int i = 0; i < 10; i++) {
        	lite.forward(input);
        	lite.getOutput().showDM("output");
        	lite.back(delta);
        	lite.diff.showDM("dx");
        }
    }
    
}

