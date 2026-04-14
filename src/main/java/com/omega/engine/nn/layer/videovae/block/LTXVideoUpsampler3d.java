package com.omega.engine.nn.layer.videovae.block;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * LTXVideoUpsampler3d
 *
 * @author Administrator
 */
public class LTXVideoUpsampler3d extends Layer {

	public int depth;
	public int oDepth;
	
	private int[] stride;
	
	private int upscale_factor;
	
//	private boolean residual = false;
	
	private boolean is_causal;
	
    public LTXVideoCausalConv3d conv;
    
    private Tensor tmp_out;

    public LTXVideoUpsampler3d(int channel, int depth, int height, int width, int[] stride, int upscale_factor, boolean residual, boolean is_causal, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.stride = stride;
        this.upscale_factor = upscale_factor;
//        this.residual = residual;
        this.is_causal = is_causal;
        initLayers();
    }

    public void initLayers() {
    	
    	int oChannel = (channel * stride[0] * stride[1] * stride[2]) / upscale_factor;

    	conv = new LTXVideoCausalConv3d(channel, oChannel, depth, width, height, 3, 3, 3, 1, true, is_causal, network);
    	conv.setUpdater(UpdaterFactory.create(this.network));
    	conv.paramsInit = ParamsInit.silu;
        
        this.oDepth = depth * stride[0] - 1;
        this.oChannel = channel;
		this.oHeight = height * stride[1];
		this.oWidth = width * stride[2];
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(output == null || output.number != number) {
        	tmp_out = Tensor.createGPUTensor(tmp_out, number, channel * depth * stride[0], height * stride[1], width * stride[2], true);
        	output = Tensor.createGPUTensor(output, number, channel * (depth * stride[0] - 1), height * stride[1], width * stride[2], true);
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
    	conv.forward(input);
    	int[] x_shape = new int[]{number, channel, stride[0], stride[1], stride[2], depth, height, width};
    	int[] o_shape = new int[]{number, channel, depth, stride[0], height, stride[1], width, stride[2]};
    	Tensor_OP().permute(conv.getOutput(), tmp_out, x_shape, o_shape, new int[] {0, 1, 5, 2, 6, 3, 7, 4});
    	Tensor_OP().getByChannel(tmp_out, output, new int[] {number * channel, depth * stride[0], height * stride[1], width * stride[2]}, stride[0] - 1, depth * stride[0] - 1);
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

