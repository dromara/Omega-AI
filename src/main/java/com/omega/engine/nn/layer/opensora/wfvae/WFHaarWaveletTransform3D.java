package com.omega.engine.nn.layer.opensora.wfvae;

import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import jcuda.runtime.JCuda;

public class WFHaarWaveletTransform3D extends Layer {
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

    private WFKernel wfKernel;

    private Tensor[] inputs = new Tensor[8];

    public WFHaarWaveletTransform3D(int channel, int depth, int width, int height) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public WFHaarWaveletTransform3D(int channel, int depth, int width, int height, Network network) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.network = network;
        initLayers();
    }

    public static void main(String[] args) {
        try {
            Thread.sleep(10000);
        } catch (Exception e) {

        }
        int N = 2;
        int C = 3;
        int F = 17;
        int H = 256;
        int W = 256;

        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C * F, H, W, data, true);

//        String inputPath = "c:\\temp\\input_wf.json";
//        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//        Tensor input = new Tensor(N, C * F, H, W, true);
//        ClipModelUtils.loadData(input, datas, "x", 5);

        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //nt channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
        WFHaarWaveletTransform3D conv1 = new WFHaarWaveletTransform3D(C,  F, W, H, nn);

        conv1.forward(input);
//        float[] delta_data = MatrixUtils.val(conv1.getOutput().dataLength, 1.0f);
//        Tensor delta = new Tensor(N, conv1.oChannel * conv1.oDepth, conv1.oHeight, conv1.oWidth, delta_data, true);
//        conv1.back(delta);
        conv1.getOutput().showShape();
        conv1.getOutput().showDM();
    }

    private void initLayers() {
        int kernelNum = 1;
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

        this.hConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.hConv.setUpdater(UpdaterFactory.create(this.network));
        this.hConv.weight.setData(h);

        this.gConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.gConv.setUpdater(UpdaterFactory.create(this.network));
        this.gConv.weight.setData(g);

        this.hhConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.hhConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhConv.weight.setData(hh);

        this.ghConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.ghConv.setUpdater(UpdaterFactory.create(this.network));
        this.ghConv.weight.setData(gh);

        this.hVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.hVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hVConv.weight.setData(hV);

        this.gVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.gVConv.setUpdater(UpdaterFactory.create(this.network));
        this.gVConv.weight.setData(gV);

        this.hhVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
        this.hhVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhVConv.weight.setData(hhV);

        this.ghVConv = new WFCausalConv3D(1, 1, depth, width, height, kernelSize, stride, false, this.network);
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
            this.output = Tensor.createGPUTensor(this.output, number, this.channel * 8 * this.ghVConv.oDepth, this.ghVConv.oHeight, this.ghVConv.oWidth, true);
        }
    }

    @Override
    public void initBack() {

    }

    @Override
    public void initParam() {

    }

    @Override
    public void output() {
        Tensor reshaped = this.input.view(this.number * this.channel, this.depth, this.height, this.width);

//        long start = System.currentTimeMillis();
        hConv.forward(reshaped);
        int length = hConv.getOutput().dataLength;
        wfKernel.append(hConv.getOutput(), this.output, length, 0);

        gConv.forward(reshaped, hConv.getOutput());
        wfKernel.append(gConv.getOutput(), this.output, length, length);

        hhConv.forward(reshaped, gConv.getOutput());
        wfKernel.append(hhConv.getOutput(), this.output, length, 2 * length);

        ghConv.forward(reshaped, hhConv.getOutput());
        wfKernel.append(ghConv.getOutput(), this.output, length, 3 * length);

        hVConv.forward(reshaped, ghConv.getOutput());
        wfKernel.append(hVConv.getOutput(), this.output, length, 4 * length);

        gVConv.forward(reshaped, hVConv.getOutput());
        wfKernel.append(gVConv.getOutput(), this.output, length, 5 * length);

        hhVConv.forward(reshaped, gVConv.getOutput());
        wfKernel.append(hhVConv.getOutput(), this.output, length, 6 * length);

        ghVConv.forward(reshaped, hhVConv.getOutput());
        wfKernel.append(ghVConv.getOutput(), this.output, length, 7 * length);

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
        wfKernel.cat_number_expend(inputs, output, ghVConv.oDepth * ghVConv.oWidth * ghVConv.oHeight);
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

    }

    @Override
    public void forward() {

    }

    @Override
    public void back() {

    }

    @Override
    public void backTemp() {

    }

    @Override
    public void back(Tensor delta) {

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
}
