package com.omega.engine.nn.layer.dc_ae;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DCAE_EncoderStages
 *
 * @author Administrator
 */
public class DCAEEncoder extends Layer {
	
	private int num_blocks = 2;
    private int group = 32;
    private int initChannel;
    private int latent_channels;
    private int num_stages;
    
    private ConvolutionLayer conv;
    private GNLayer norm;
    private SiLULayer act;
    private DCAEResidual res1;
    private DCAEResidual res2;
    
    private List<DCAEEncoderStage> encoders;
    
    private ConvolutionLayer midConv;
    private GNLayer midNorm;
    private SiLULayer midAct;
    
    private List<DCAEResidual> midResList;
    
    private ConvolutionLayer midOut;
    
    public DCAEEncoder(int channel, int initChannel, int height, int width, int group, int num_blocks,int num_stages,int latent_channels, Network network) {
        this.network = network;
        this.channel = channel;
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
    	
    	conv = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, this.network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
        norm = new GNLayer(group, conv.oChannel, conv.oHeight, conv.oWidth, BNType.conv_bn, conv);
        act = new SiLULayer(norm);
    	
        int ih = conv.oHeight;
        int iw = conv.oWidth;
        res1 = new DCAEResidual(conv.oChannel, initChannel, ih, iw, group, network);
        ih = res1.oHeight;
        iw = res1.oWidth;
        res2 = new DCAEResidual(initChannel, initChannel, ih, iw, group, network);
        
        encoders = new ArrayList<DCAEEncoderStage>();
        
        int ic = initChannel;
    	for(int i = 0;i<num_stages;i++) {
    		int oc = ic * 2;
    		DCAEEncoderStage encoder = new DCAEEncoderStage(ic, oc, ih, iw, group, num_blocks, network);
    		encoders.add(encoder);
    		ic = encoder.oChannel;
    		ih = encoder.oHeight;
    		iw = encoder.oWidth;
    	}
    	
    	midConv = new ConvolutionLayer(ic, ic * 2, iw, ih,  1, 1, 0, 1, true, this.network);
    	midConv.setUpdater(UpdaterFactory.create(this.network));
    	midConv.paramsInit = ParamsInit.silu;
    	ic = midConv.oChannel;
		ih = midConv.oHeight;
		iw = midConv.oWidth;
    	midNorm = new GNLayer(group, midConv.oChannel, midConv.oHeight, midConv.oWidth, BNType.conv_bn, midConv);
    	midAct = new SiLULayer(midNorm);
    	
    	midResList = new ArrayList<DCAEResidual>();
    	for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = new DCAEResidual(ic, ic, ih, iw, group, network);
    		midResList.add(res);
    		ic = res.oChannel;
    		ih = res.oHeight;
    		iw = res.oWidth;
    	}
    	
    	midOut = new ConvolutionLayer(ic, latent_channels, iw, ih,  1, 1, 0, 1, true, this.network);
    	midOut.setUpdater(UpdaterFactory.create(this.network));
    	midOut.paramsInit = ParamsInit.silu;
    	
        this.oHeight = midOut.oHeight;
        this.oWidth = midOut.oWidth;
        this.oChannel = midOut.oChannel;
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
    	
    	conv.forward(input);
    	norm.forward(conv.getOutput());
    	act.forward(norm.getOutput());

    	res1.forward(act.getOutput());
    	res2.forward(res1.getOutput());
    	
    	Tensor x = res2.getOutput();
    	for(int i = 0;i<num_stages;i++) {
    		DCAEEncoderStage encoder = encoders.get(i);
    		encoder.forward(x);
    		x = encoder.getOutput();
    	}
    	
    	/**
    	 * midden
    	 */
    	midConv.forward(x);
    	midNorm.forward(midConv.getOutput());
    	midAct.forward(midNorm.getOutput());
    	
    	Tensor mx = midAct.getOutput();
    	for(int i = 0;i<num_blocks;i++) {
    		DCAEResidual res = midResList.get(i);
    		res.forward(mx);
    		mx = res.getOutput();
    	}
    	
    	midOut.forward(mx);
    	
        this.output = midOut.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    	midOut.back(delta);
    	
    	Tensor mdy = midOut.diff;
    	for(int i = num_blocks-1;i>=0;i--) {
    		DCAEResidual res = midResList.get(i);
    		res.back(mdy);
    		mdy = res.diff;
    	}

    	midAct.back(mdy);
    	midNorm.back(midAct.diff);
    	midConv.back(midNorm.diff);
    	
    	Tensor dy = midConv.diff;
    	for(int i = num_stages - 1;i>=0;i--) {
    		DCAEEncoderStage encoder = encoders.get(i);
    		encoder.back(dy);
    		dy = encoder.diff;
    	}
    	
    	res2.back(dy);
    	res1.back(res2.diff);
    	
    	act.back(res1.diff);
    	norm.back(act.diff);
    	conv.back(norm.diff);
        this.diff = conv.diff;
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
    	norm.update();
    	res1.update();
    	res2.update();
    	for(int i = 0;i<num_stages;i++) {
    		encoders.get(i).update();
    	}
    	midConv.update();
    	midNorm.update();

    	for(int i = 0;i<num_blocks;i++) {
    		midResList.get(i).update();
    	}
    	
    	midOut.update();
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
    	norm.saveModel(outputStream);
    	res1.saveModel(outputStream);
    	res2.saveModel(outputStream);
    	for(int i = 0;i<num_stages;i++) {
    		encoders.get(i).saveModel(outputStream);
    	}
    	midConv.saveModel(outputStream);
    	midNorm.saveModel(outputStream);

    	for(int i = 0;i<num_blocks;i++) {
    		midResList.get(i).saveModel(outputStream);
    	}
    	
    	midOut.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
    	norm.loadModel(inputStream);
    	res1.loadModel(inputStream);
    	res2.loadModel(inputStream);
    	for(int i = 0;i<num_stages;i++) {
    		encoders.get(i).loadModel(inputStream);
    	}
    	midConv.loadModel(inputStream);
    	midNorm.loadModel(inputStream);

    	for(int i = 0;i<num_blocks;i++) {
    		midResList.get(i).loadModel(inputStream);
    	}
    	
    	midOut.loadModel(inputStream);
    }
}

