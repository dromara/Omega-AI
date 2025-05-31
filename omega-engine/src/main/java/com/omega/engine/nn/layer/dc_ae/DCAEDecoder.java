package com.omega.engine.nn.layer.dc_ae;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DCAE Decoder
 *
 * @author Administrator
 */
public class DCAEDecoder extends Layer {
	
	private int num_blocks = 2;
    private int group = 32;
    private int initChannel;
    private int latent_channels;
    private int num_stages;
    
    private List<DCAEDecoderStage> decoders;
    
    private DCAEResidual finalRes1;
    private DCAEResidual finalRes2;
    
    private ConvolutionLayer out;
    
    public DCAEDecoder(int channel, int initChannel, int height, int width, int group, int num_blocks,int num_stages,int latent_channels, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = initChannel;
        this.initChannel = initChannel;
        this.height = height;
        this.width = width;
        this.group = group;
        this.num_blocks= num_blocks;
        this.num_stages= num_stages;
        this.latent_channels = latent_channels;
        initLayers(initChannel);
    }

    public void initLayers(int oChannel) {
    	
    	decoders = new ArrayList<DCAEDecoderStage>();
    	int ic = latent_channels;
    	int ih = height;
    	int iw = width;
    	for(int i = 0;i<num_stages;i++) {
    		int oc = Math.max(ic/2, initChannel/2);
    		DCAEDecoderStage decoder = new DCAEDecoderStage(ic, oc, ih, iw, group, num_blocks, network);
    		decoders.add(decoder);
    		ic = decoder.oChannel;
    		ih = decoder.oHeight;
    		iw = decoder.oWidth;
    	}

    	finalRes1 = new DCAEResidual(ic, ic, ih, iw, group, network);
        ih = finalRes1.oHeight;
        iw = finalRes1.oWidth;
        finalRes2 = new DCAEResidual(ic, ic, ih, iw, group, network);
        ih = finalRes2.oHeight;
        iw = finalRes2.oWidth;
        
    	out = new ConvolutionLayer(ic, channel, iw, ih,  3, 3, 1, 1, true, this.network);
    	out.setUpdater(UpdaterFactory.create(this.network));
    	
        this.oHeight = out.oHeight;
        this.oWidth = out.oWidth;
        this.oChannel = out.oChannel;
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
    	Tensor x = input;
    	for(int i = 0;i<num_stages;i++) {
    		DCAEDecoderStage decoder = decoders.get(i);
    		decoder.forward(x);
    		x = decoder.getOutput();
    	}
    	
    	finalRes1.forward(x);
    	finalRes2.forward(finalRes1.getOutput());
    	
    	out.forward(finalRes2.getOutput());
    	
        this.output = out.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    	out.back(delta);

    	finalRes2.back(out.diff);
    	finalRes1.back(finalRes2.diff);

    	Tensor dy = finalRes1.diff;
    	for(int i = num_stages - 1;i>=0;i--) {
    		DCAEDecoderStage decoder = decoders.get(i);
    		decoder.back(dy);
    		dy = decoder.diff;
    	}
    	
        this.diff = dy;
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
    	for(int i = 0;i<num_stages;i++) {
    		decoders.get(i).update();
    	}
    	finalRes1.update();
    	finalRes2.update();
    	out.update();
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
    	for(int i = 0;i<num_stages;i++) {
    		decoders.get(i).saveModel(outputStream);
    	}
    	finalRes1.saveModel(outputStream);
    	finalRes2.saveModel(outputStream);
    	out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	for(int i = 0;i<num_stages;i++) {
    		decoders.get(i).loadModel(inputStream);
    	}
    	finalRes1.loadModel(inputStream);
    	finalRes2.loadModel(inputStream);
    	out.loadModel(inputStream);
    }
}

