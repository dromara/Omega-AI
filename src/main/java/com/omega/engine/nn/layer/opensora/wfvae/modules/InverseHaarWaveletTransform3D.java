package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Convolution3DTransposeLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.runtime.JCuda;

public class InverseHaarWaveletTransform3D extends Layer {
	
    private Convolution3DTransposeLayer hConv;
    private Convolution3DTransposeLayer gConv;
    private Convolution3DTransposeLayer hhConv;
    private Convolution3DTransposeLayer ghConv;
    private Convolution3DTransposeLayer hVConv;
    private Convolution3DTransposeLayer gVConv;
    private Convolution3DTransposeLayer hhVConv;
    private Convolution3DTransposeLayer ghVConv;

    private int channel;
    private int depth;

    private Tensor tmp;
    private Tensor d_tmp;
    
    private WFKernel wfKernel;

    public int oDepth;

    public InverseHaarWaveletTransform3D(int channel, int depth, int width, int height) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        initLayers();
        this.oDepth = hConv.oDepth - 1;
        this.oChannel = channel / 8;
        this.oHeight = hConv.oHeight;
        this.oWidth = hConv.oWidth;
    }

    public InverseHaarWaveletTransform3D(int channel, int depth, int width, int height, Network network) {
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.network = network;
        initLayers();
        this.oDepth = hConv.oDepth - 1;
        this.oChannel = channel / 8;
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
        int C = 24;
        int F = 5;
        int H = 128;
        int W = 128;
//
//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);

        String inputPath = "D:\\models\\input_wf.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(input, datas, "x", 5);

        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //nt channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
        InverseHaarWaveletTransform3D conv1 = new InverseHaarWaveletTransform3D(C, F, W, H, nn);
        
        String deltaPath = "D:\\models\\delta_wf.json";
        Map<String, Object> delta_datas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, 3 * 9, H * 2, W * 2, true);
        ClipModelUtils.loadData(delta, delta_datas, "delta", 5);
        
        for(int i = 0;i<10;i++) {
        	long start = System.nanoTime();
        	conv1.forward(input);
        	conv1.back(delta);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start)/1e6+"ms.");
        }        
        
//        float[] delta_data = MatrixUtils.order(conv1.getOutput().dataLength, 0.001f, 0.001f);
//        Tensor delta = new Tensor(N, conv1.oChannel * conv1.oDepth, conv1.oHeight, conv1.oWidth, delta_data, true);

        delta.showShape();
        
//        conv1.getOutput().showShape();
        conv1.getOutput().showDM();
        conv1.diff.showDM("diff:");
//        conv1.getOutput().showDMByNumber(1);
    }

    private void initLayers() {

        float[] h = createWeight(new float[]{1, 1, 1, 1, 1, 1, 1, 1}, 0.3536f);
        float[] g = createWeight(new float[]{1, -1, 1, -1, 1, -1, 1, -1}, 0.3536f);
        float[] hh = createWeight(new float[]{1, 1, -1, -1, 1, 1, -1, -1}, 0.3536f);
        float[] gh = createWeight(new float[]{1, -1, -1, 1, 1, -1, -1, 1}, 0.3536f);
        float[] hV = createWeight(new float[]{1, 1, 1, 1, -1, -1, -1, -1}, 0.3536f);
        float[] gV = createWeight(new float[]{1, -1, 1, -1, -1, 1, -1, 1}, 0.3536f);
        float[] hhV = createWeight(new float[]{1, 1, -1, -1, -1, -1, 1, 1}, 0.3536f);
        float[] ghV = createWeight(new float[]{1, -1, -1, 1, -1, 1, 1, -1}, 0.3536f);

        this.hConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.hConv.setUpdater(UpdaterFactory.create(this.network));
        this.hConv.weight.setData(h);

        this.gConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.gConv.setUpdater(UpdaterFactory.create(this.network));
        this.gConv.weight.setData(g);

        this.hhConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.hhConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhConv.weight.setData(hh);

        this.ghConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.ghConv.setUpdater(UpdaterFactory.create(this.network));
        this.ghConv.weight.setData(gh);

        this.hVConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.hVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hVConv.weight.setData(hV);

        this.gVConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.gVConv.setUpdater(UpdaterFactory.create(this.network));
        this.gVConv.weight.setData(gV);

        this.hhVConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
        this.hhVConv.setUpdater(UpdaterFactory.create(this.network));
        this.hhVConv.weight.setData(hhV);

        this.ghVConv = new Convolution3DTransposeLayer(1, 1, depth, width, height, 2, 2, 2, 0, 2, 1, 0, false, true, network);
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
    public void init() {
        this.number = this.network.number;
        if (this.output == null || this.number != this.output.number) {
        	this.tmp = Tensor.createGPUTensor(this.tmp, number * oChannel, this.depth, height, width, true);
            this.output = Tensor.createGPUTensor(this.output, number, oChannel * oDepth, this.hConv.oHeight, this.hConv.oWidth, true);
        }
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if (this.output == null || this.number != this.output.number) {
        	this.tmp = Tensor.createGPUTensor(this.tmp, number * oChannel, this.depth, height, width, true);
            this.output = Tensor.createGPUTensor(this.output, number, oChannel * oDepth, this.hConv.oHeight, this.hConv.oWidth, true);
        }else {
        	this.output.clearGPU();
        }
    }

    @Override
    public void initBack() {
    	if(this.diff == null || this.diff.number != number) {
    		diff = Tensor.createGPUTensor(diff, input.shape(), true);
    		d_tmp = Tensor.createGPUTensor(d_tmp, hConv.getOutput().shape(), true);
    	}
    }

    @Override
    public void initParam() {

    }

    @Override
    public void output() {
    	
    	input.view(number, channel, depth * height, width);

    	Tensor_OP().getByChannel(input, tmp, 0, oChannel);    	
        hConv.forward(tmp);
        wfKernel.add_offset(hConv.getOutput(), output, hConv.oDepth - 1, hConv.oHeight * hConv.oWidth, 1);
        
        Tensor_OP().getByChannel(input, tmp, 1 * oChannel, oChannel);
        gConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(gConv.getOutput(), output, gConv.oDepth - 1, gConv.oHeight * gConv.oWidth, 1);
        
        Tensor_OP().getByChannel(input, tmp, 2 * oChannel, oChannel);
        hhConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(hhConv.getOutput(), output, hhConv.oDepth - 1, hhConv.oHeight * hhConv.oWidth, 1);

        Tensor_OP().getByChannel(input, tmp, 3 * oChannel, oChannel);
        ghConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(ghConv.getOutput(), output, ghConv.oDepth - 1, ghConv.oHeight * ghConv.oWidth, 1);

        Tensor_OP().getByChannel(input, tmp, 4 * oChannel, oChannel);
        hVConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(hVConv.getOutput(), output, hVConv.oDepth - 1, hVConv.oHeight * hVConv.oWidth, 1);
       
        Tensor_OP().getByChannel(input, tmp, 5 * oChannel, oChannel);
        gVConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(gVConv.getOutput(), output, gVConv.oDepth - 1, gVConv.oHeight * gVConv.oWidth, 1);

        Tensor_OP().getByChannel(input, tmp, 6 * oChannel, oChannel);
        hhVConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(hhVConv.getOutput(), output, hhVConv.oDepth - 1, hhVConv.oHeight * hhVConv.oWidth, 1);

        Tensor_OP().getByChannel(input, tmp, 7 * oChannel, oChannel);
        ghVConv.forward(tmp, hConv.getOutput());
        wfKernel.add_offset(ghVConv.getOutput(), output, ghVConv.oDepth - 1, ghVConv.oHeight * ghVConv.oWidth, 1);
        
        input.viewOrg();
    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {
    	
    	diff.view(number, channel, depth * height, width);
    	
    	Tensor_OP().getByChannel_back(d_tmp, delta, 1, oDepth);
    	
    	ghVConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 7 * oChannel, oChannel);

    	hhVConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 6 * oChannel, oChannel);
    	
    	gVConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 5 * oChannel, oChannel);
    	
    	hVConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 4 * oChannel, oChannel);
    	
    	ghConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 3 * oChannel, oChannel);
    	
    	hhConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 2 * oChannel, oChannel);
    	
    	gConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 1 * oChannel, oChannel);
    	
    	hConv.back(d_tmp, tmp);
    	Tensor_OP().getByChannel_back(diff, tmp, 0 * oChannel, oChannel);
    	
    	diff.viewOrg();
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
}
