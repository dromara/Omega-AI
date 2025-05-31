package com.omega.engine.nn.layer.dc_ae;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * DCAE_EncoderStages
 *
 * @author Administrator
 */
public class DCAEEncoderStage extends Layer {
	
	private int num_blocks = 2;
    private int group = 32;
    private DCAEResidualDownsampleBlock down;
    private List<DCAEResidual> resList; 
    
    public DCAEEncoderStage(int channel, int oChannel, int height, int width, int group, int num_blocks, Network network) {
        this.network = network;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.group = group;
        this.num_blocks= num_blocks;
        initLayers(oChannel);
    }

    public void initLayers(int oChannel) {
    	
    	down = new DCAEResidualDownsampleBlock(channel, oChannel, height, width, group, network);
    	
    	resList = new ArrayList<DCAEResidual>();
    	
    	int ih = down.oHeight;
    	int iw = down.oWidth;
    	for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = new DCAEResidual(oChannel, oChannel, ih, iw, group, network);
    		resList.add(res);
    		ih = res.oHeight;
    		iw = res.oWidth;
    	}
    	
        this.oHeight = ih;
        this.oWidth = iw;
        this.oChannel = oChannel;
    }

    @Override
    public void init() {
        this.number = this.network.number;

    }
    
    public void init(Tensor input) {
        this.number = input.number;
        
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
    	
    	down.forward(input);
    	
    	Tensor x = down.getOutput();
    	for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = resList.get(i);
    		res.forward(x);
    		x = res.getOutput();
    	}
    	
        this.output = x;

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor dy = delta;
    	for(int i = num_blocks - 1;i>=0;i--) {
    		DCAEResidual res = resList.get(i);
    		res.back(dy);
    		dy = res.diff;
    	}
    	down.back(dy);
        this.diff = down.diff;
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
        down.update();
        for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = resList.get(i);
    		res.update();
    	}
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
        down.saveModel(outputStream);
        for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = resList.get(i);
    		res.saveModel(outputStream);
    	}
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        down.loadModel(inputStream);
        for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = resList.get(i);
    		res.loadModel(inputStream);
    	}
    }
}

