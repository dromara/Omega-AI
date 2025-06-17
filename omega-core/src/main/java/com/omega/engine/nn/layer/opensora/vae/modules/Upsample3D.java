package com.omega.engine.nn.layer.opensora.vae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.UPSample3DLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * Downsample2D
 *
 * @author Administrator
 */
public class Upsample3D extends Layer {

    private int scale_factor;
    
    public int depth;
    public int oDepth;
    
    public UPSample3DLayer up;
    
    public Res3DBlockUpsample conv3d;
    
    public Upsample3D(int channel,int depth, int height, int width,int scale_factor, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.scale_factor = scale_factor;
        initLayers();
        this.oChannel = conv3d.oChannel;
        this.oDepth = conv3d.oDepth;
        this.oHeight = conv3d.oHeight;
        this.oWidth = conv3d.oWidth;
    }

    public void initLayers() {
    	
    	up = new UPSample3DLayer(channel, depth, height, width, scale_factor, network);
        
    	conv3d = new Res3DBlockUpsample(up.oChannel, up.oDepth, up.oHeight, up.oWidth, network);

    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(output == null || output.getShape()[0] != this.number) {
        	output = Tensor.createGPUTensor(output, number, conv3d.oChannel * conv3d.oDepth, conv3d.oHeight, conv3d.oWidth, true);
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
        
    	up.forward(input);
    	
    	conv3d.forward(up.getOutput());
    	
    	Tensor_OP().add(conv3d.getOutput(), up.getOutput(), output);

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println(index);
    	conv3d.back(delta);
    	
    	Tensor_OP().add(conv3d.diff, delta, conv3d.diff);
    	
    	up.back(conv3d.diff);
    	
        this.diff = up.diff;

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
        conv3d.update();
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
    	conv3d.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv3d.loadModel(inputStream);
    }
}

