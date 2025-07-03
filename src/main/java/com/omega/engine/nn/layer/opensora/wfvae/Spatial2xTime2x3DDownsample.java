package com.omega.engine.nn.layer.opensora.wfvae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Spatial2xTime2x3DDownsample
 *
 * @author Administrator
 */
public class Spatial2xTime2x3DDownsample extends Layer {

	public PaddingKernel kernel;

    public WFCausalConv3D conv;

    private Tensor padOutput;
    
    private int[] padding = new int[] {0, 1, 0, 1, 0, 0};
    
    private int pd;
    private int pw;
    private int ph;
    
    public int depth;
    
    public int oDepth;
    
    public Spatial2xTime2x3DDownsample(int channel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
        this.oChannel = conv.oChannel;
        this.oDepth = conv.oDepth;
    }

    public void initLayers() {
    	
        kernel = new PaddingKernel(cuda());
        
        pd = padding[4] + depth + padding[5];
        pw = padding[0] + width + padding[1];
        ph = padding[2] + height + padding[3];
        
        conv = new WFCausalConv3D(channel, oChannel, pd, pw, ph, 3, 2, hasBias, network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(padOutput == null || padOutput.number != this.number) {
        	padOutput = Tensor.createGPUTensor(padOutput, number, channel * pd, ph, pw, true);
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
        // TODO Auto-generated method stub
    	kernel.padding3d(input, padOutput, pd, padding, 0);
    	conv.forward(padOutput);
    	this.output = conv.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        conv.back(delta, padOutput);
        kernel.padding3dGrad(padOutput, input, depth, padding);
        this.diff = input;
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
        conv.update();
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
        conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
    }
}

