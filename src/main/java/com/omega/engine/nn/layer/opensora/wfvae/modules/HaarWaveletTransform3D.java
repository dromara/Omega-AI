package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.runtime.JCuda;

public class HaarWaveletTransform3D extends Layer {
	
    private WFCausalConv3D hConv;
    private WFCausalConv3D gConv;
    private WFCausalConv3D hhConv;
    private WFCausalConv3D ghConv;
    private WFCausalConv3D hVConv;
    private WFCausalConv3D gVConv;
    private WFCausalConv3D hhVConv;
    private WFCausalConv3D ghVConv;

    private int channel;
    private int depth;
    public int oDepth;

    private WFKernel wfKernel;

//    private Tensor tmp;
    private Tensor out_kc;

//    private Tensor[] inputs = new Tensor[8];

    public HaarWaveletTransform3D(int channel, int depth, int width, int height) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oDepth = hConv.oDepth;
        this.oChannel = channel;
        this.oHeight = hConv.oHeight;
        this.oWidth = hConv.oWidth;
    }

    public HaarWaveletTransform3D(int channel, int depth, int width, int height, Network network) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.network = network;
        initLayers();
        this.oDepth = hConv.oDepth;
        this.oChannel = channel;
        this.oHeight = hConv.oHeight;
        this.oWidth = hConv.oWidth;
    }

    public static void main(String[] args) {
//        try {
//            Thread.sleep(10000);
//        } catch (Exception e) {
//
//        }
        int N = 2;
        int C = 3;
        int F = 9;
        int H = 32;
        int W = 32;
//
//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);

        String inputPath = "D:\\models\\input_hw3d.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(input, datas, "x", 5);

        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //nt channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
        HaarWaveletTransform3D conv1 = new HaarWaveletTransform3D(C,  F, W, H, nn);

        String deltaPath = "D:\\models\\delta_hw3d.json";
        Map<String, Object> delta_datas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, 24 * 5, H / 2, W / 2, true);
        ClipModelUtils.loadData(delta, delta_datas, "delta", 5);

//        conv1.forward(input);
//        conv1.getOutput().showDMByOffsetRed(3 * 5 * 16 * 16, 5 * 16 * 16, "1");
//        conv1.back(delta);
//        conv1.diff.showDM();
        for(int i = 0;i<10;i++) {
        	long start = System.nanoTime();
        	conv1.forward(input);
        	conv1.back(delta);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start)/1e6+"ms.");
        }
        conv1.getOutput().showDM();
        conv1.diff.showDM();
//        float[] delta_data = MatrixUtils.val(conv1.getOutput().dataLength, 1.0f);
//        Tensor delta = new Tensor(N, conv1.oChannel * conv1.oDepth, conv1.oHeight, conv1.oWidth, delta_data, true);
//        conv1.back(delta);
//        conv1.getOutput().showShape();
//        conv1.getOutput().showDMByNumber(1);
    }

    private void initLayers() {
        int kernelSize = 2;
        int stride = 2;
        float[] h = createWeight(new float[]{1, 1, 1, 1, 1, 1, 1, 1}, 0.3536f);
        float[] g = createWeight(new float[]{1, -1, 1, -1, 1, -1, 1, -1}, 0.3536f);
        float[] hh = createWeight(new float[]{1, 1, -1, -1, 1, 1, -1, -1}, 0.3536f);
        float[] gh = createWeight(new float[]{1, -1, -1, 1, 1, -1, -1, 1}, 0.3536f);
        float[] hV = createWeight(new float[]{1, 1, 1, 1, -1, -1, -1, -1}, 0.3536f);
        float[] gV = createWeight(new float[]{1, -1, 1, -1, -1, 1, -1, 1}, 0.3536f);
        float[] hhV = createWeight(new float[]{1, 1, -1, -1, -1, -1, 1, 1}, 0.3536f);
        float[] ghV = createWeight(new float[]{1, -1, -1, 1, -1, 1, 1, -1}, 0.3536f);

        this.hConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.hConv.setUpdater(UpdaterFactory.create(this.network));
        this.hConv.weight.setData(h);

        this.gConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.gConv.setUpdater(UpdaterFactory.create(this.network));
        this.gConv.weight.setData(g);

        this.hhConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.hhConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhConv.weight.setData(hh);

        this.ghConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.ghConv.setUpdater(UpdaterFactory.create(this.network));
        this.ghConv.weight.setData(gh);

        this.hVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.hVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hVConv.weight.setData(hV);

        this.gVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.gVConv.setUpdater(UpdaterFactory.create(this.network));
        this.gVConv.weight.setData(gV);

        this.hhVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.hhVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhVConv.weight.setData(hhV);

        this.ghVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, true, this.network);
        this.ghVConv.setUpdater(UpdaterFactory.create(this.network));
        this.ghVConv.weight.setData(ghV);

        if(wfKernel == null) {
            wfKernel = new WFKernel(this.cuda());
        }
    }

    private float[] createWeight(float[] values, float scale) {
        float[] kernel = new float[8];
        for (int i = 0; i < 8; i++) {
            kernel[i] = values[i] * scale;
        }
        return kernel;
    }

    @Override
    public void forward(Tensor input) {

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
    public void init() {
        this.number = this.network.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, 8 * this.channel * this.ghVConv.oDepth, this.ghVConv.oHeight, this.ghVConv.oWidth, true);
            out_kc = Tensor.createGPUTensor(this.out_kc, number, this.channel * 8 * this.ghVConv.oDepth, this.ghVConv.oHeight, this.ghVConv.oWidth, true);
        }
    }

    @Override
    public void initBack() {
        if(this.diff == null || this.diff.number != number) {
            diff = Tensor.createGPUTensor(diff, input.shape(), true);
        }else {
            diff.clearGPU();
        }
    }

    @Override
    public void initParam() {

    }

    @Override
    public void output() {
        Tensor reshaped = this.input.view(this.number * this.channel, this.depth, this.height, this.width);

//        long start = System.currentTimeMillis();
        hConv.forward(reshaped);

        int length = hConv.oChannel * hConv.oDepth * hConv.oWidth * hConv.oHeight;

        wfKernel.append(hConv.getOutput(), this.out_kc, length, 8, 0);

        gConv.forward(reshaped, hConv.getOutput());
        wfKernel.append(gConv.getOutput(), this.out_kc, length, 8, 1);

        hhConv.forward(reshaped, gConv.getOutput());
        wfKernel.append(hhConv.getOutput(), this.out_kc, length, 8, 2);

        ghConv.forward(reshaped, hhConv.getOutput());
        wfKernel.append(ghConv.getOutput(), this.out_kc, length, 8, 3);

        hVConv.forward(reshaped, ghConv.getOutput());
        wfKernel.append(hVConv.getOutput(), this.out_kc, length, 8, 4);

        gVConv.forward(reshaped, hVConv.getOutput());
        wfKernel.append(gVConv.getOutput(), this.out_kc, length, 8, 5);

        hhVConv.forward(reshaped, gVConv.getOutput());
        wfKernel.append(hhVConv.getOutput(), this.out_kc, length, 8, 6);

        ghVConv.forward(reshaped, hhVConv.getOutput());
        wfKernel.append(ghVConv.getOutput(), this.out_kc, length, 8, 7);

        Tensor_OP().permute(out_kc, output, new int[] {number, channel, 8, ghVConv.oDepth, ghVConv.oHeight, ghVConv.oWidth},
        		new int[] {number, 8, channel, ghVConv.oDepth, ghVConv.oHeight, ghVConv.oWidth}, new int[] {0, 2, 1, 3, 4, 5});
        
        this.input.viewOrg();
        
//        JCuda.cudaDeviceSynchronize();
//        long end = System.currentTimeMillis();
//        System.out.println(end - start);


//        long start = System.currentTimeMillis();
//        hConv.forward(reshaped);
//        gConv.forward(reshaped);
//        hhConv.forward(reshaped);
//        ghConv.forward(reshaped);
//        hVConv.forward(reshaped);
//        gVConv.forward(reshaped);
//        hhVConv.forward(reshaped);
//        ghVConv.forward(reshaped);
//
//        inputs[0] = hConv.getOutput();
//        inputs[1] = gConv.getOutput();
//        inputs[2] = hhConv.getOutput();
//        inputs[3] = ghConv.getOutput();
//        inputs[4] = hVConv.getOutput();
//        inputs[5] = gVConv.getOutput();
//        inputs[6] = hhVConv.getOutput();
//        inputs[7] = ghVConv.getOutput();
//
//        ghVConv.getOutput().syncHost();
//
//        wfKernel.cat_number_expend(inputs, output, ghVConv.oDepth * ghVConv.oWidth * ghVConv.oHeight);
//        JCuda.cudaDeviceSynchronize();
//        long end = System.currentTimeMillis();
//        System.out.println(end - start);

    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {
    	
        Tensor_OP().permute(delta, out_kc, new int[] {number, 8, channel, ghVConv.oDepth, ghVConv.oHeight, ghVConv.oWidth},
        		new int[] {number, channel, 8, ghVConv.oDepth, ghVConv.oHeight, ghVConv.oWidth}, new int[] {0, 2, 1, 3, 4, 5});
    	
        Tensor tmp = ghVConv.getOutput();
        
        int[] kc_shape = new int[] {number * channel, 8, 1, ghVConv.oDepth * ghVConv.oHeight * ghVConv.oWidth};
        
        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 7, 1);
        ghVConv.back(tmp);
        Tensor_OP().add(diff, ghVConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 6, 1);
        hhVConv.back(tmp, ghVConv.diff);
        Tensor_OP().add(diff, hhVConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 5, 1);
        gVConv.back(tmp, hhVConv.diff);
        Tensor_OP().add(diff, gVConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 4, 1);
        hVConv.back(tmp, gVConv.diff);
        Tensor_OP().add(diff, hVConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 3, 1);
        ghConv.back(tmp, hVConv.diff);
        Tensor_OP().add(diff, ghConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 2, 1);
        hhConv.back(tmp, ghConv.diff);
        Tensor_OP().add(diff, hhConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 1, 1);
        gConv.back(tmp, hhConv.diff);
        Tensor_OP().add(diff, gConv.diff, diff);

        Tensor_OP().getByChannel(out_kc, tmp, kc_shape, 0, 1);
        hConv.back(tmp, gConv.diff);
        Tensor_OP().add(diff, hConv.diff, diff);

    }

    @Override
    public void forward() {

    }

    @Override
    public void back() {
        initBack();
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
    public void backTemp() {

    }

    @Override
    public void back(Tensor delta) {
        initBack();
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

    }

    @Override
    public void accGrad(float scale) {

    }

    @Override
    public void showDiff() {

    }

    @Override
    public LayerType getLayerType() {
        return null;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        return new float[0][][][];
    }

    @Override
    public void initCache() {

    }
    
    public void saveModel(RandomAccessFile outputStream) throws IOException {
    	
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	
    }
    
}
