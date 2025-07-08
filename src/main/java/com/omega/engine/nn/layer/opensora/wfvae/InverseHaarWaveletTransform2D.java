package com.omega.engine.nn.layer.opensora.wfvae;

import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;


public class InverseHaarWaveletTransform2D extends Layer {

    public ConvolutionTransposeLayer conv_ll;
    public ConvolutionTransposeLayer conv_lh;
    public ConvolutionTransposeLayer conv_hl;
    public ConvolutionTransposeLayer conv_hh;

    private int depth;

    private Tensor inputT;
    private Tensor outputT;

    private WFKernel wfKernel;

    public InverseHaarWaveletTransform2D(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv_hh.oHeight;
        this.oWidth = conv_hh.oWidth;
    }

    public void initLayers() {
    	conv_ll = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, true, this.network);
    	conv_ll.setUpdater(UpdaterFactory.create(this.network));
    	conv_ll.paramsInit = ParamsInit.silu;
    	conv_ll.weight.setData(new float[] {0.5f, 0.5f, 0.5f, 0.5f});
    	
    	conv_lh = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, true, this.network);
    	conv_lh.setUpdater(UpdaterFactory.create(this.network));
    	conv_lh.paramsInit = ParamsInit.silu;
    	conv_lh.weight.setData(new float[] {0.5f, 0.5f, -0.5f, -0.5f});
    	
    	conv_hl = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, true, this.network);
    	conv_hl.setUpdater(UpdaterFactory.create(this.network));
    	conv_hl.paramsInit = ParamsInit.silu;
    	conv_hl.weight.setData(new float[] {0.5f, -0.5f, 0.5f, -0.5f});
    	
    	conv_hh = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, true, this.network);
    	conv_hh.setUpdater(UpdaterFactory.create(this.network));
    	conv_hh.paramsInit = ParamsInit.silu;
    	conv_hh.weight.setData(new float[] {0.5f, -0.5f, -0.5f, 0.5f});
    	
    	if(wfKernel == null) {
    		wfKernel = new WFKernel(this.cuda());
    	}
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(inputT == null || inputT.number != this.number * depth) {
        	inputT = Tensor.createGPUTensor(inputT, number * depth * channel, 1, height, width, true);
        	this.outputT = Tensor.createGPUTensor(output, number, depth * channel, conv_hh.oHeight, conv_hh.oWidth, true);
        	this.output = Tensor.createGPUTensor(output, number, channel * depth, conv_hh.oHeight, conv_hh.oWidth, true);
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
        Tensor lowLow = new Tensor(number * depth, channel, height, width, true);
        wfKernel.chunk(input, lowLow, width * height, input.dataLength / 4, 4, 0);
        lowLow.showShape();
    	Tensor_OP().permute(lowLow, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        inputT.showShape();
    	conv_ll.forward(inputT);
        conv_ll.getOutput().syncHost();
        lowLow.syncHost();
        inputT.syncHost();
        Tensor_OP().add(outputT, conv_ll.getOutput(), outputT);

        Tensor lowHigh = new Tensor(number * depth, channel, height, width, true);
        wfKernel.chunk(input, lowHigh, width * height, input.dataLength / 4, 4, 1);
        Tensor_OP().permute(lowHigh, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        conv_lh.forward(inputT);
        Tensor_OP().add(outputT, conv_lh.getOutput(), outputT);

        Tensor highLow = new Tensor(number * depth, channel, height, width, true);
        wfKernel.chunk(input, highLow, width * height, input.dataLength / 4, 4, 2);
        Tensor_OP().permute(highLow, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        conv_hl.forward(inputT);
        Tensor_OP().add(outputT, conv_hl.getOutput(), outputT);

        Tensor highHigh = new Tensor(number * depth, channel, height, width, true);
        wfKernel.chunk(input, highHigh, width * height, input.dataLength / 4, 4, 3);
        Tensor_OP().permute(highHigh, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        conv_hh.forward(inputT);
        Tensor_OP().add(outputT, conv_hh.getOutput(), outputT);

        outputT.view(number, channel * depth, height * 2, width * 2);

    	Tensor_OP().permute(outputT, output, new int[] {number, depth, channel, conv_hh.oHeight, conv_hh.oWidth}, new int[] {number, channel, depth, conv_hh.oHeight, conv_hh.oWidth}, new int[]{0, 2, 1, 3, 4});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

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
        int F = 17;
        int H = 32;
        int W = 32;

//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);

        String inputPath = "c:\\temp\\input_wf.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	Tensor input = new Tensor(N, C * F, H, W, true);
    	ClipModelUtils.loadData(input, datas, "x", 5);
        
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        HaarWaveletTransform2D t2d = new HaarWaveletTransform2D(C, F, H, W, nn);
//
        t2d.forward(input);
        
//        t2d.output.showDMByNumber(1);
//        t2d.output.showShape();

        InverseHaarWaveletTransform2D tr2d = new InverseHaarWaveletTransform2D(C, F, H/2, W/2, nn);
        tr2d.forward(t2d.output);
        tr2d.output.showShape();
        tr2d.output.showDM();
        tr2d.output.showDMByNumber(0);
    }
    
}

