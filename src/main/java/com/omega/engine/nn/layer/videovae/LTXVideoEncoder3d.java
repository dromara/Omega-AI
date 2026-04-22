package com.omega.engine.nn.layer.videovae;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.layer.videovae.block.LTXVideoCausalConv3d;
import com.omega.engine.nn.layer.videovae.block.LTXVideoDownBlock3D;
import com.omega.engine.nn.layer.videovae.block.LTXVideoMidBlock3d;
import com.omega.engine.nn.layer.videovae.kernel.LTXVideoVAEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * LTXVideoEncoder3d
 *
 * @author Administrator
 */
public class LTXVideoEncoder3d extends Layer {

	private int batchSize;
	
	public int depth;
	public int oDepth;
	
	private int patch_size;
	private int patch_size_t;
	
	private int[] block_out_channels = new int[] {128, 256, 512, 512};
	
	private boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
	
	private int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
	
	private boolean is_causal;
	
    public LTXVideoCausalConv3d conv_in;
    
    public List<LTXVideoDownBlock3D> down_blocks;
    
    public LTXVideoMidBlock3d mid_block;
    
    public RMSLayer norm_out;
    private SiLULayer conv_act;
    public LTXVideoCausalConv3d conv_out;
    
    private Tensor video_x;
    
    private Tensor normInput;
    
    private LTXVideoVAEKernel kernel;

    public LTXVideoEncoder3d(int channel, int oChannel, int depth, int height, int width, int patch_size, int patch_size_t, int[] block_out_channels, int[] layers_per_block, boolean[] spatio_temporal_scaling, boolean is_causal, Network network) {
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
    	
    	kernel = new LTXVideoVAEKernel(cuda());
    	
       	int inChannel = block_out_channels[0];
    	
       	//batchSize, channel, patch_size_t, patch_size, patch_size, post_patch_num_frames, post_patch_height, post_patch_width
    	int post_patch_num_frames = depth / patch_size_t;
    	int post_patch_height = height / patch_size;
    	int post_patch_width = width / patch_size;
       	
    	conv_in = new LTXVideoCausalConv3d(channel * (patch_size * patch_size), inChannel, post_patch_num_frames, post_patch_width, post_patch_height, 3, 3, 3, 1, true, is_causal, network);
//	    System.err.println("conv_in.oDepth:"+conv_in.oDepth);
//	    System.err.println("conv_in.oHeight:"+conv_in.oHeight);
//	    System.err.println("conv_in.oWidth:"+conv_in.oWidth);
    	down_blocks = new ArrayList<LTXVideoDownBlock3D>();
    	
    	int inDepth = conv_in.oDepth;
    	int inHeight = conv_in.oHeight;
     	int inWidth = conv_in.oWidth;
    	for(int i = 0;i<block_out_channels.length;i++) {
    		int oc = block_out_channels[i];
    		if(i + 1 < block_out_channels.length) {
    			oc = block_out_channels[i+1];
    		}
    		LTXVideoDownBlock3D block = new LTXVideoDownBlock3D(inChannel, oc, inDepth, inHeight, inWidth, layers_per_block[i], is_causal, spatio_temporal_scaling[i], network);
    		down_blocks.add(block);
    		inDepth = block.oDepth;
    		inChannel = oc;
    		inHeight = block.oHeight;
    		inWidth = block.oWidth;
    	}
    	
    	mid_block = new LTXVideoMidBlock3d(inChannel, inChannel, inDepth, inHeight, inWidth, layers_per_block[layers_per_block.length-1], is_causal, network);
    	norm_out = new RMSLayer(1, 1, inChannel, false, BNType.fully_bn, network);
    	conv_act = new SiLULayer(norm_out);
    	conv_out = new LTXVideoCausalConv3d(inChannel, oChannel+1, inDepth, inWidth, inHeight, 3, 3, 3, 1, true, is_causal, network);
//        System.err.println("conv_out.oDepth:"+conv_out.oDepth);
//        System.err.println("conv_out.oHeight:"+conv_out.oHeight);
//        System.err.println("conv_out.oWidth:"+conv_out.oWidth);
        this.oDepth = conv_out.oDepth;
		this.oHeight = conv_out.oHeight;
		this.oWidth = conv_out.oWidth;
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(video_x == null || video_x.number != number) {
        	int post_patch_num_frames = depth / patch_size_t;
        	int post_patch_height = height / patch_size;
        	int post_patch_width = width / patch_size;
        	//batchSize, channel, patch_size_t, patch_size, patch_size, post_patch_num_frames, post_patch_height, post_patch_width
        	video_x = Tensor.createGPUTensor(video_x, number, channel * patch_size_t * patch_size * patch_size * post_patch_num_frames, post_patch_height, post_patch_width, true);
        	int inChannel = mid_block.oChannel;
        	int inDepth = mid_block.oDepth;
        	int inHeight = mid_block.oHeight;
        	int inWidth = mid_block.oWidth;
        	normInput = Tensor.createGPUTensor(normInput, number, inDepth, inHeight * inWidth, inChannel, true);
        	output = Tensor.createGPUTensor(output, number, oChannel * 2 * oDepth, oHeight, oWidth, true);
        }else {
        	normInput.viewOrg();
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
    	//batch_size, num_channels, num_frames, height, width = hidden_states.shape;
        
    	//post_patch_num_frames = num_frames // p_t
    	//post_patch_height = height // p
    	//post_patch_width = width // p
    	int post_patch_num_frames = depth / patch_size_t;
    	int post_patch_height = height / patch_size;
    	int post_patch_width = width / patch_size;

    	//batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
    	int[] x_shape = new int[] {batchSize, channel, post_patch_num_frames, patch_size_t, post_patch_height, patch_size , post_patch_width, patch_size};
    	int[] v_shape = new int[] {batchSize, channel, patch_size_t, patch_size, patch_size, post_patch_num_frames, post_patch_height, post_patch_width};
    	
    	Tensor_OP().permute(input, video_x, x_shape, v_shape, new int[] {0, 1, 3, 7, 5, 2, 4, 6});
//    	video_x.showDMByOffsetRed(((3 * 17) + 16) * post_patch_height * post_patch_width, post_patch_height * post_patch_width, "video_x");
    	conv_in.forward(video_x);
    	
    	Tensor x = conv_in.getOutput();
//    	x.showDMByOffsetRed(0, post_patch_height * post_patch_width, "conv_in");
    	for(int i = 0;i<down_blocks.size();i++) {
    		LTXVideoDownBlock3D down = down_blocks.get(i);
    		down.forward(x);
    		x = down.getOutput();
//    		x.showShape(i+"");
//    		x.showDMByOffsetRed((3 * down.oDepth + 2) * x.height * x.width, x.height * x.width, i+"");
    	}
    	
    	mid_block.forward(x);
//    	mid_block.getOutput().showDMByOffsetRed((3 * mid_block.oDepth + 2) * x.height * x.width, x.height * x.width, "mid_block");
    	
    	int inChannel = mid_block.oChannel;
    	int inDepth = mid_block.oDepth;
    	int inHeight = mid_block.oHeight;
    	int inWidth = mid_block.oWidth;
    	Tensor_OP().permute(mid_block.getOutput(), normInput, new int[] {number, inChannel, inDepth, inHeight, inWidth}, new int[] {number, inDepth, inHeight, inWidth, inChannel}, new int[] {0, 2, 3, 4, 1});
    	norm_out.forward(normInput);
        Tensor_OP().permute(norm_out.getOutput(), normInput, new int[] {number, inDepth, inHeight, inWidth, inChannel}, new int[] {number, inChannel, inDepth, inHeight, inWidth}, new int[] {0, 4, 1, 2, 3});
        normInput.view(number, inChannel * inDepth, inHeight, inWidth);
        conv_act.forward(normInput);
        conv_out.forward(conv_act.getOutput());
//        conv_out.getOutput().showDMByOffsetRed((3 * 3 + 2) * conv_out.getOutput().height * conv_out.getOutput().width, conv_out.getOutput().height * conv_out.getOutput().width, "conv_out.getOutput()");

        kernel.encoder_repeat(conv_out.getOutput(), output, oChannel * 2, oDepth, oHeight, oWidth, oChannel, oChannel-1);
//        output.showShape("output");
//        output.showDMByOffsetRed((255 * 3 + 2) * output.height * output.width, output.height * output.width, "output");
        
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
        for (int i = 0; i < down_blocks.size(); i++) {
        	LTXVideoDownBlock3D l = down_blocks.get(i);
            l.saveModel(outputStream);
        }
        mid_block.saveModel(outputStream);
//        norm_out.saveModel(outputStream);
        conv_out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv_in.loadModel(inputStream);
        for (int i = 0; i < down_blocks.size(); i++) {
        	LTXVideoDownBlock3D l = down_blocks.get(i);
            l.loadModel(inputStream);
        }
        mid_block.loadModel(inputStream);
//        norm_out.loadModel(inputStream);
        conv_out.loadModel(inputStream);
    }
}

