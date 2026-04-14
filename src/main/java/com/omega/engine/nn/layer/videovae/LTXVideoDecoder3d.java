package com.omega.engine.nn.layer.videovae;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.layer.videovae.block.LTXVideoCausalConv3d;
import com.omega.engine.nn.layer.videovae.block.LTXVideoMidBlock3d;
import com.omega.engine.nn.layer.videovae.block.LTXVideoUpBlock3d;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * LTXVideoDecoder3d
 *
 * @author Administrator
 */
public class LTXVideoDecoder3d extends Layer {

	public int depth;
	public int oDepth;
	
	private int patch_size;
	private int patch_size_t;
	
	private int upsample_factor = 1;
	
	private int[] block_out_channels = new int[] {512, 512, 256, 128};
	
	private boolean[] spatio_temporal_scaling = new boolean[] {false, true, true, true};
	
	private int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
	
	private boolean is_causal;
	
    public LTXVideoCausalConv3d conv_in;
    
    public LTXVideoMidBlock3d mid_block;
    
    public List<LTXVideoUpBlock3d> up_blocks;

    public RMSLayer norm_out;
    private SiLULayer conv_act;
    public LTXVideoCausalConv3d conv_out;
    
    private Tensor normInput;
    
    public LTXVideoDecoder3d(int channel, int oChannel, int depth, int height, int width, int patch_size, int patch_size_t, int[] block_out_channels, int[] layers_per_block, boolean[] spatio_temporal_scaling, boolean is_causal, Network network) {
    	this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.patch_size = patch_size;
        this.patch_size_t = patch_size_t;
        this.block_out_channels = block_out_channels;
        this.spatio_temporal_scaling = spatio_temporal_scaling;
        this.layers_per_block = layers_per_block;
        this.is_causal = is_causal;
        initLayers();
    }

    public void initLayers() {
       	int inChannel = channel;
    	int output_channel = block_out_channels[0];
//    	System.err.println(JsonUtils.toJson(block_out_channels));
//    	System.err.println(inChannel+":"+output_channel);
    	conv_in = new LTXVideoCausalConv3d(inChannel, output_channel, depth, width, height, 3, 3, 3, 1, true, is_causal, network);

    	mid_block = new LTXVideoMidBlock3d(output_channel, output_channel, conv_in.oDepth, conv_in.oHeight, conv_in.oWidth, layers_per_block[0], is_causal, network);

    	up_blocks = new ArrayList<LTXVideoUpBlock3d>();
    	
    	int inDepth = depth;
    	int inHeight = height;
     	int inWidth = width;
    	for(int i = 0;i<block_out_channels.length;i++) {
    		int input_channel = output_channel / upsample_factor;
    		output_channel = block_out_channels[i] / upsample_factor;
    		LTXVideoUpBlock3d block = new LTXVideoUpBlock3d(input_channel, output_channel, inDepth, inHeight, inWidth, layers_per_block[i+1], upsample_factor, is_causal, spatio_temporal_scaling[i], network);
    		up_blocks.add(block);
    		inDepth = block.oDepth;
    		inChannel = output_channel;
    		inHeight = block.oHeight;
    		inWidth = block.oWidth;
    	}
        
    	norm_out = new RMSLayer(1, 1, inChannel, false, BNType.fully_bn, network);
    	conv_act = new SiLULayer(norm_out);
    	//self.out_channels = out_channels * patch_size**2

    	conv_out = new LTXVideoCausalConv3d(inChannel, oChannel * (patch_size * patch_size), inDepth, inWidth, inHeight, 3, 3, 3, 1, true, is_causal, network);
    	
        this.oDepth = conv_out.oDepth * patch_size_t;
		this.oHeight = conv_out.oHeight * patch_size;
		this.oWidth = conv_out.oWidth * patch_size;
		
//        System.err.println("decoder.oDepth:"+this.oDepth);
//        System.err.println("decoder.oHeight:"+this.oHeight);
//        System.err.println("decoder.oWidth:"+this.oWidth);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(output == null || output.number != number) {
        	normInput = Tensor.createGPUTensor(normInput, number, up_blocks.get(up_blocks.size() - 1).oDepth, up_blocks.get(up_blocks.size() - 1).oHeight * up_blocks.get(up_blocks.size() - 1).oWidth, up_blocks.get(up_blocks.size() - 1).oChannel, true);
        	output = Tensor.createGPUTensor(output, number, oChannel * oDepth, oHeight, oWidth, true);
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
    	conv_in.forward(input);

    	conv_in.getOutput().showDMByOffsetRed((3 * 3 + 2) * conv_in.getOutput().height * conv_in.getOutput().width, conv_in.getOutput().height * conv_in.getOutput().width, "conv_in.getOutput()");
    	
    	mid_block.forward(conv_in.getOutput());
    	
    	Tensor x = mid_block.getOutput();
    	x.showDMByOffsetRed((3 * mid_block.oDepth + 2) * x.height * x.width, x.height * x.width, "mid_block");
    	for(int i = 0;i<up_blocks.size();i++) {
    		LTXVideoUpBlock3d up = up_blocks.get(i);
    		up.forward(x);
    		x = up.getOutput();
        	x.showDMByOffsetRed((3 * up.oDepth + 2) * x.height * x.width, x.height * x.width, i+"");
    	}
    	
    	int inDepth = up_blocks.get(up_blocks.size() - 1).oDepth;
    	int inChannel = up_blocks.get(up_blocks.size() - 1).oChannel;
    	int inHeight = up_blocks.get(up_blocks.size() - 1).oHeight;
    	int inWidth = up_blocks.get(up_blocks.size() - 1).oWidth;
    	Tensor_OP().permute(x, normInput, new int[] {number, inChannel, inDepth, inHeight, inWidth}, new int[] {number, inDepth, inHeight, inWidth, inChannel}, new int[] {0, 2, 3, 4, 1});
    	norm_out.forward(normInput);
        Tensor_OP().permute(norm_out.getOutput(), normInput, new int[] {number, inDepth, inHeight, inWidth, inChannel}, new int[] {number, inChannel, inDepth, inHeight, inWidth}, new int[] {0, 4, 1, 2, 3});
    	conv_act.forward(normInput);
        conv_out.forward(conv_act.getOutput());
    	
        int[] x_shape = new int[] {number, oChannel, patch_size_t, patch_size, patch_size, conv_out.oDepth, conv_out.oHeight, conv_out.oWidth};
        int[] o_shape = new int[] {number, oChannel, conv_out.oDepth, patch_size_t, conv_out.oHeight, patch_size, conv_out.oWidth, patch_size};
        Tensor_OP().permute(conv_out.getOutput(), output, x_shape, o_shape, new int[] {0, 1, 5, 2, 6, 4, 7, 3});
//        output.showShape("output");
        output.showDMByOffsetRed((2 * 17 + 2) * output.height * output.width, output.height * output.width, "output");
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
    	conv_in.saveModel(outputStream);
    	mid_block.saveModel(outputStream);
    	for(int i = 0;i<up_blocks.size();i++) {
    		LTXVideoUpBlock3d block = up_blocks.get(i);
    		block.saveModel(outputStream);
    	}
//    	norm_out.saveModel(outputStream);
    	conv_out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv_in.loadModel(inputStream);
    	mid_block.loadModel(inputStream);
    	for(int i = 0;i<up_blocks.size();i++) {
    		LTXVideoUpBlock3d block = up_blocks.get(i);
    		block.loadModel(inputStream);
    	}
//    	norm_out.loadModel(inputStream);
    	conv_out.loadModel(inputStream);
    }
    
}

