package com.omega.engine.nn.layer.sd_vae.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * VQVAEUpsample
 *
 * @author Administrator
 */
public class SDVAEDownsample extends Layer {

    public ConvolutionLayer conv;
    
    private PaddingKernel kernel;
    
    private int[] padding = new int[] {0, 1, 0, 1};
    
    private Tensor pout;
    
    private int ph;
    private int pw;
    
    public SDVAEDownsample(int channel, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
    }

    public void initLayers() {
    	
    	this.ph = height + padding[0] + padding[1];
    	this.pw = width + padding[2] + padding[3];
    	
        conv = new ConvolutionLayer(channel, oChannel, pw, ph, 3, 3, 0, 2, true, this.network);
        
        if(kernel == null) {
        	kernel = new PaddingKernel(cuda());
        }
        
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(pout == null || pout.number != this.number) {
        	pout = Tensor.createGPUTensor(pout, number, channel, ph, pw, true);
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
    	kernel.padding2d(input, pout, padding, 0);
        conv.forward(pout);
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

