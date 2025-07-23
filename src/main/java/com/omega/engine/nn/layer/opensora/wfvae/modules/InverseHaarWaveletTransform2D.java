package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;


public class InverseHaarWaveletTransform2D extends Layer {

    public ConvolutionTransposeLayer conv_ll;
    public ConvolutionTransposeLayer conv_lh;
    public ConvolutionTransposeLayer conv_hl;
    public ConvolutionTransposeLayer conv_hh;

    public int depth;
    public int oDepth;

    private Tensor inputT;
    private Tensor outputT;
    private Tensor tmp;

    private WFKernel wfKernel;

    public InverseHaarWaveletTransform2D(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = channel / 4;
        this.height = height;
        this.width = width;
        initLayers();
        this.oDepth = depth;
        this.oHeight = conv_hh.oHeight;
        this.oWidth = conv_hh.oWidth;
    }

    public void initLayers() {
        conv_ll = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, false, this.network);
        conv_ll.setUpdater(UpdaterFactory.create(this.network));
        conv_ll.paramsInit = ParamsInit.silu;
        conv_ll.weight.setData(new float[] {0.5f, 0.5f, 0.5f, 0.5f});

        conv_lh = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, false, this.network);
        conv_lh.setUpdater(UpdaterFactory.create(this.network));
        conv_lh.paramsInit = ParamsInit.silu;
        conv_lh.weight.setData(new float[] {0.5f, 0.5f, -0.5f, -0.5f});

        conv_hl = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, false, this.network);
        conv_hl.setUpdater(UpdaterFactory.create(this.network));
        conv_hl.paramsInit = ParamsInit.silu;
        conv_hl.weight.setData(new float[] {0.5f, -0.5f, 0.5f, -0.5f});

        conv_hh = new ConvolutionTransposeLayer(1, 1, width, height, 2, 2, 0, 2, 1, 0, false, this.network);
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
            inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
            this.tmp = Tensor.createGPUTensor(this.tmp, number * depth * oChannel, 1, height, width, true);
            this.outputT = Tensor.createGPUTensor(output, number,  depth * oChannel, conv_hh.oHeight, conv_hh.oWidth, true);
            this.output = Tensor.createGPUTensor(output, number, oChannel * depth, conv_hh.oHeight, conv_hh.oWidth, true);
        }else {
        	outputT.clearGPU();
        }
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if(inputT == null || inputT.number != this.number * depth) {
            inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
            this.tmp = Tensor.createGPUTensor(this.tmp, number * depth * oChannel, 1, height, width, true);
            this.outputT = Tensor.createGPUTensor(output, number,  depth * oChannel, conv_hh.oHeight, conv_hh.oWidth, true);
            this.output = Tensor.createGPUTensor(output, number, oChannel * depth, conv_hh.oHeight, conv_hh.oWidth, true);
        }else {
        	outputT.clearGPU();
        }
    }

    @Override
    public void initBack() {
        if(this.diff == null || this.diff.number != number) {
            diff = Tensor.createGPUTensor(diff, input.shape(), true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        Tensor_OP().permute(input, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        int len = oChannel;

        int[] shape = new int[] {number * depth, channel, height, width};
        
        Tensor_OP().getByChannel(inputT, tmp, shape, 0);
        conv_ll.forward(tmp);
        Tensor_OP().add(outputT, conv_ll.getOutput(), outputT);

        Tensor_OP().getByChannel(inputT, tmp, shape, len);
        conv_lh.forward(tmp, conv_ll.getOutput());
        Tensor_OP().add(outputT, conv_lh.getOutput(), outputT);

        Tensor_OP().getByChannel(inputT, tmp, shape, 2 * len);
        conv_hl.forward(tmp, conv_lh.getOutput());
        Tensor_OP().add(outputT, conv_hl.getOutput(), outputT);

        Tensor_OP().getByChannel(inputT, tmp, shape, 3 * len);
        conv_hh.forward(tmp, conv_hl.getOutput());
        Tensor_OP().add(outputT, conv_hh.getOutput(), outputT);

        Tensor_OP().permute(outputT, output, new int[] {number, depth, oChannel, conv_hh.oHeight, conv_hh.oWidth}, new int[] {number, oChannel, depth, conv_hh.oHeight, conv_hh.oWidth}, new int[]{0, 2, 1, 3, 4});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        Tensor_OP().permute(delta, outputT, new int[] {number, oChannel, depth, conv_hh.oHeight, conv_hh.oWidth}, new int[] {number, depth, oChannel, conv_hh.oHeight, conv_hh.oWidth}, new int[]{0, 2, 1, 3, 4});

        int[] shape = new int[] {number * depth, channel, height, width};
        
        conv_hh.back(outputT, tmp);
        Tensor_OP().getByChannel_back(inputT, tmp, shape, 3 * oChannel);

        conv_hl.back(outputT, tmp);
        Tensor_OP().getByChannel_back(inputT, tmp, shape, 2 * oChannel);

        conv_lh.back(outputT, tmp);
        Tensor_OP().getByChannel_back(inputT, tmp, shape, oChannel);

        conv_ll.back(outputT, tmp);
        Tensor_OP().getByChannel_back(inputT, tmp, shape, 0);

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
        int C = 4;
        int F = 17;
        int H = 32;
        int W = 32;

//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F * 4, H, W, data, true);

        String inputPath = "c:\\temp\\input_wf.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(input, datas, "x", 5);

        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;

        InverseHaarWaveletTransform2D tr2d = new InverseHaarWaveletTransform2D(C, F, H, W, nn);
        tr2d.forward(input);
        tr2d.output.showShape();
        tr2d.output.showDM();

        String deltaPath = "c:\\temp\\delta1_wf.json";
        Map<String, Object> delta_datas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, 1 * F, H * 2, W * 2, true);
        ClipModelUtils.loadData(delta, delta_datas, "delta", 5);

        tr2d.back(delta);
    }

}

