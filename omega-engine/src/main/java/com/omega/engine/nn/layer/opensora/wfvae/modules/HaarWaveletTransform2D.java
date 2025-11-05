package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.PrintUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * HaarWaveletTransform2D
 * video to image
 * @author Administrator
 */
public class HaarWaveletTransform2D extends Layer {

    public ConvolutionLayer conv_ll;
    public ConvolutionLayer conv_lh;
    public ConvolutionLayer conv_hl;
    public ConvolutionLayer conv_hh;

    public int depth;
    public int oDepth;
    
    private Tensor inputT;
    private Tensor outputT;
    
//    private Tensor[] inputs = new Tensor[4];
    
    public HaarWaveletTransform2D(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oDepth = depth;
        this.oHeight = conv_hh.oHeight;
        this.oWidth = conv_hh.oWidth;
    }

    public void initLayers() {
    	conv_ll = new ConvolutionLayer(1, 1, width, height, 2, 2, 0, 2, false, true, this.network);
    	conv_ll.setUpdater(UpdaterFactory.create(this.network));
    	conv_ll.paramsInit = ParamsInit.silu;
    	conv_ll.weight.setData(new float[] {0.5f, 0.5f, 0.5f, 0.5f});
    	
    	conv_lh = new ConvolutionLayer(1, 1, width, height, 2, 2, 0, 2, false, true, this.network);
    	conv_lh.setUpdater(UpdaterFactory.create(this.network));
    	conv_lh.paramsInit = ParamsInit.silu;
    	conv_lh.weight.setData(new float[] {0.5f, 0.5f, -0.5f, -0.5f});
    	
    	conv_hl = new ConvolutionLayer(1, 1, width, height, 2, 2, 0, 2, false, true, this.network);
    	conv_hl.setUpdater(UpdaterFactory.create(this.network));
    	conv_hl.paramsInit = ParamsInit.silu;
    	conv_hl.weight.setData(new float[] {0.5f, -0.5f, 0.5f, -0.5f});
    	
    	conv_hh = new ConvolutionLayer(1, 1, width, height, 2, 2, 0, 2, false, true, this.network);
    	conv_hh.setUpdater(UpdaterFactory.create(this.network));
    	conv_hh.paramsInit = ParamsInit.silu;
    	conv_hh.weight.setData(new float[] {0.5f, -0.5f, -0.5f, 0.5f});

    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(inputT == null || inputT.number != this.number * depth) {
        	inputT = Tensor.createGPUTensor(inputT, number * depth * channel, 1, height, width, true);
        	this.outputT = Tensor.createGPUTensor(output, number, depth * 4 * channel, conv_hh.oHeight, conv_hh.oWidth, true);
        	this.output = Tensor.createGPUTensor(output, number, 4 * channel * depth, conv_hh.oHeight, conv_hh.oWidth, true);
        }
    }

    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.number != number) {
            diff = Tensor.createGPUTensor(diff, input.shape(), true);
        }
    	inputT.clearGPU();
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(input, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});

    	conv_ll.forward(inputT);

    	int[] shape = new int[] {number * depth, 4, 1, channel * conv_ll.oHeight * conv_ll.oWidth};
    	
   	 	Tensor_OP().getByChannel_back(outputT, conv_ll.getOutput(), shape, 0, 1);

    	conv_lh.forward(inputT, conv_ll.getOutput());
   	 	Tensor_OP().getByChannel_back(outputT, conv_ll.getOutput(), shape, 1, 1);

    	conv_hl.forward(inputT, conv_ll.getOutput());
   	 	Tensor_OP().getByChannel_back(outputT, conv_ll.getOutput(), shape, 2, 1);

    	conv_hh.forward(inputT, conv_ll.getOutput());
   	 	Tensor_OP().getByChannel_back(outputT, conv_ll.getOutput(), shape, 3, 1);

    	Tensor_OP().permute(outputT, output, new int[] {number, depth, 4, channel, conv_hh.oHeight, conv_hh.oWidth}, new int[] {number, 4, channel, depth, conv_hh.oHeight, conv_hh.oWidth}, new int[]{0, 2, 3, 1, 4, 5});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, outputT, new int[] {number, 4, channel, depth, conv_hh.oHeight, conv_hh.oWidth}, new int[] {number, depth, 4, channel, conv_hh.oHeight, conv_hh.oWidth}, new int[]{0, 3, 1, 2, 4, 5});
    	
    	int[] shape = new int[] {number * depth, 4, 1, channel * conv_ll.oHeight * conv_ll.oWidth};
    	
        Tensor tmp = conv_hh.getOutput();
        
        Tensor_OP().getByChannel(outputT, tmp, shape, 3, 1);
        conv_hh.back(tmp);
        Tensor_OP().add(inputT, conv_hh.diff, inputT);
        
        Tensor_OP().getByChannel(outputT, tmp, shape, 2, 1);
        conv_hl.back(tmp, conv_hh.diff);
        Tensor_OP().add(inputT, conv_hl.diff, inputT);
        
        Tensor_OP().getByChannel(outputT, tmp, shape, 1, 1);
        conv_lh.back(tmp, conv_hh.diff);
        Tensor_OP().add(inputT, conv_lh.diff, inputT);
        
        Tensor_OP().getByChannel(outputT, tmp, shape, 0, 1);
        conv_ll.back(tmp, conv_hh.diff);
        Tensor_OP().add(inputT, conv_ll.diff, inputT);
        
        Tensor_OP().permute(inputT, diff, new int[] {number, depth, channel, height, width}, new int[] {number, channel, depth, height, width}, new int[]{0, 2, 1, 3, 4});
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

    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {

    }
    
    public static void main(String args[]) {
    	int N = 2;
        int C = 3;
        int F = 9;
        int H = 4;
        int W = 4;

//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);

        String inputPath = "D:\\models\\input_hw3d.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	Tensor input = new Tensor(N, C * F, H, W, true);
    	ModeLoaderlUtils.loadData(input, datas, "x", 5);
        
    	String deltaPath = "D:\\models\\delta_hw3d.json";
        Map<String, Object> delta_datas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, 108, H / 2, W / 2, true);
        ModeLoaderlUtils.loadData(delta, delta_datas, "delta", 5);
    	
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        HaarWaveletTransform2D t2d = new HaarWaveletTransform2D(C, F, H, W, nn);
        
        t2d.forward(input);
        t2d.getOutput().showShape();
        PrintUtils.printImage(t2d.getOutput());
        t2d.getOutput().showDMByOffsetRed(3 * 9 * 2 * 2, 9 * 2 * 2, "r[0][4]");
        
        t2d.back(delta);
        
        t2d.diff.showDM();
        
//        t2d.output.showDMByNumber(1);
//        t2d.output.showShape();
        
    }
    
}

