package com.omega.engine.nn.layer.opensora.vae;

import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.opensora.vae.modules.AttentionBlock3D;
import com.omega.engine.nn.layer.opensora.vae.modules.CausalConv3DPlainAR;
import com.omega.engine.nn.layer.opensora.vae.modules.GNLayer3D;
import com.omega.engine.nn.layer.opensora.vae.modules.Resnet3DBlock;
import com.omega.engine.nn.layer.opensora.vae.modules.Upsample2D;
import com.omega.engine.nn.layer.opensora.vae.modules.Upsample3D;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * VideoDecoder
 *
 * @author Administrator
 */
public class VideoDecoder extends Layer {
	
	public int depth;
	public int oDepth;
	
    private int num_res_blocks;
    private int ch;
    private int[] ch_mult = new int[] {1, 2, 4, 4};
    private int z_channels;
    private int[] temporal_up_layer = new int[] {2, 3};
    
    private int temporal_downsample = 4;
    
    public CausalConv3DPlainAR convIn;
    public List<Layer> upBlock;
    public List<Layer> midBlock;
    public GNLayer3D convNormOut;
    public SiLULayer convAct;
    public CausalConv3DPlainAR convOut;

    public VideoDecoder(int z_channels, int oChannel, int depth, int height, int width, int ch, int num_res_blocks, int[] ch_mult, int[] temporal_up_layer,int temporal_downsample, Network network) {
        this.network = network;
        this.channel = z_channels;
        this.z_channels = z_channels;
        this.oChannel = oChannel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.ch = ch;
        this.num_res_blocks = num_res_blocks;
        this.ch_mult = ch_mult;
        this.temporal_up_layer = temporal_up_layer;
        this.temporal_downsample = temporal_downsample;
        initLayers();
    }

    public void initLayers() {
    	
    	List<Integer> temporal_up_layers = new ArrayList<Integer>();
    	
    	for(int idx:temporal_up_layer) {
    		temporal_up_layers.add(idx);
    	}
    	int block_in = ch * ch_mult[ch_mult.length - 1];
        convIn = new CausalConv3DPlainAR(z_channels, block_in, depth, width, height, 3, 1, true, network);
        convIn.setUpdater(UpdaterFactory.create(this.network));
        convIn.paramsInit = ParamsInit.silu;
        upBlock = new ArrayList<Layer>();
        int outc = block_in;
        int id = convIn.oDepth;
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        
        // mid
        midBlock = new ArrayList<Layer>();
        Resnet3DBlock mb1 = new Resnet3DBlock(outc, outc, id, ih, iw, network);
        midBlock.add(mb1);
        id = mb1.oDepth;
        ih = mb1.oHeight;
        iw = mb1.oWidth;
        AttentionBlock3D attn = new AttentionBlock3D(outc, id, ih, iw, true, true, network);
        midBlock.add(attn);
        id = attn.oDepth;
        ih = attn.oHeight;
        iw = attn.oWidth;
        Resnet3DBlock mb2 = new Resnet3DBlock(outc, outc, id, ih, iw, network);
        midBlock.add(mb2);
        id = mb2.oDepth;
        ih = mb2.oHeight;
        iw = mb2.oWidth;
        
        for (int i = ch_mult.length - 1; i >= 0; i--) {
            int inc = outc;
            outc = ch * ch_mult[i];
            for(int nr = 0;nr<num_res_blocks + 1;nr++) {
            	Resnet3DBlock res3d = new Resnet3DBlock(inc, outc, id, ih, iw, network);
            	upBlock.add(res3d);
            	inc = outc;
            	id = res3d.oDepth;
                ih = res3d.oHeight;
                iw = res3d.oWidth;
            }
            if(i != 0) {
            	if(temporal_up_layers.contains(i)) {
            		Upsample3D up3d = new Upsample3D(inc, id, ih, iw, 2, network);
            		upBlock.add(up3d);
                	id = up3d.oDepth;
                    ih = up3d.oHeight;
                    iw = up3d.oWidth;
            	}else {
            		Upsample2D up2d = new Upsample2D(inc, id, ih, iw, network);
            		upBlock.add(up2d);
            		ih = up2d.oHeight;
                    iw = up2d .oWidth;
            	}
            }

        }
       
        //out
        convNormOut = new GNLayer3D(outc, id, ih, iw, 32, network);
        convAct = new SiLULayer(convNormOut);
        convOut = new CausalConv3DPlainAR(outc, oChannel, id, iw, ih, 3, 1, true, network);
        convOut.setUpdater(UpdaterFactory.create(this.network));
        convOut.paramsInit = ParamsInit.silu;
        this.oDepth = convOut.oDepth - (temporal_downsample - 1);
        this.oHeight = convOut.oHeight;
        this.oWidth = convOut.oWidth;
    }

    @Override
    public void init() {
    	this.number = network.number;
    	if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }
    
    public void init(Tensor input) {
    	this.number = input.number;
    	if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
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
    	this.input.showDM("z:");
        convIn.forward(this.input);
        
        Tensor x = convIn.getOutput();
        for (int i = 0; i < midBlock.size(); i++) {
        	Layer layer = midBlock.get(i);
            layer.forward(x);
            x = layer.getOutput();
        }

        for (int i = 0; i < upBlock.size(); i++) {
            Layer layer = upBlock.get(i);
            layer.forward(x);
            x = layer.getOutput();
        }
        convNormOut.forward(x);
        convAct.forward(convNormOut.getOutput());
        convOut.forward(convAct.getOutput());

        convOut.getOutput().view(number * convOut.oChannel, convOut.oDepth, convOut.oHeight, convOut.oWidth);

        Tensor_OP().getByPosition(convOut.getOutput(), this.output, new int[] {1 , (temporal_downsample - 1), oDepth}, 0);
        convOut.getOutput().viewOrg();
//        output.showDM("output:");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
//    	delta.showDM("delta");
    	convOut.getOutput().clearGPU();
    	convOut.getOutput().view(number * convOut.oChannel, convOut.oDepth, convOut.oHeight, convOut.oWidth);
    	Tensor_OP().getByPosition(convOut.getOutput(), delta, new int[] {1 , (temporal_downsample - 1), oDepth}, 1);
    	convOut.getOutput().viewOrg();
        convOut.back(convOut.getOutput());
//        convOut.diff.showDM("h_a_diff:");
        convAct.back(convOut.diff);
        convNormOut.back(convAct.diff);
        Tensor d = convNormOut.diff;
        d.showShape("norm");
//        d.showDM("h_block_delta:");
        for (int i = upBlock.size() - 1; i >= 0; i--) {
            Layer up = upBlock.get(i);
            up.back(d);
            d = up.diff;
        }
//        d.showDM("h_block_delta:");
        for (int i = midBlock.size() - 1; i >= 0; i--) {
            Layer mid = midBlock.get(i);
            mid.back(d);
            d = mid.diff;
//            d.showDM("mid_"+i);
        }
        convIn.back(d);
//        convIn.diff.showDM("decoder-z_delta:");
        this.diff = convIn.diff;
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
        convIn.update();
        for (int i = 0; i < midBlock.size(); i++) {
            midBlock.get(i).update();
        }
        for (int i = 0; i < upBlock.size(); i++) {
        	upBlock.get(i).update();
        }
        convNormOut.update();
        convOut.update();
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
}

