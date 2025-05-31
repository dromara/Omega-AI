package com.omega.engine.nn.layer.opensora.vae;

import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.opensora.vae.modules.AttentionBlock3D;
import com.omega.engine.nn.layer.opensora.vae.modules.CausalConv3DPlainAR;
import com.omega.engine.nn.layer.opensora.vae.modules.Downsample2D;
import com.omega.engine.nn.layer.opensora.vae.modules.Downsample3D;
import com.omega.engine.nn.layer.opensora.vae.modules.GNLayer3D;
import com.omega.engine.nn.layer.opensora.vae.modules.Resnet3DBlock;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * VideoEncoder
 *
 * @author Administrator
 */
public class VideoEncoder extends Layer {
	
	public int depth;
	public int oDepth;
	
    private int num_res_blocks;
    private int ch;
    private int[] ch_mult = new int[] {1, 2, 4, 4};
    private int z_channels;
    private int[] down_sampling_layer = new int[] {1, 2};
    
    private boolean double_z = true;
    
    public CausalConv3DPlainAR convIn;
    public List<Layer> downBlock;
    public List<Layer> midBlock;
    public GNLayer3D convNormOut;
    public SiLULayer convAct;
    public CausalConv3DPlainAR convOut;

    public VideoEncoder(int channel, int z_channels, int depth, int height, int width, int ch, int num_res_blocks, int[] ch_mult, int[] down_sampling_layer, boolean double_z, Network network) {
        this.network = network;
        this.channel = channel;
        this.z_channels = z_channels;
        this.oChannel = z_channels;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.ch = ch;
        this.num_res_blocks = num_res_blocks;
        this.ch_mult = ch_mult;
        this.down_sampling_layer = down_sampling_layer;
        this.double_z = double_z;
        initLayers();
    }

    public void initLayers() {
    	
    	List<Integer> down_sampling_layers = new ArrayList<Integer>();
    	
    	for(int idx:down_sampling_layer) {
    		down_sampling_layers.add(idx);
    	}
        convIn = new CausalConv3DPlainAR(channel, ch, depth, width, height, 3, 1, true, network);
        convIn.setUpdater(UpdaterFactory.create(this.network));
        convIn.paramsInit = ParamsInit.silu;
        downBlock = new ArrayList<Layer>();
        int outc = ch;
        int id = convIn.oDepth;
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        for (int i = 0; i < ch_mult.length; i++) {
            int inc = outc;
            outc = ch * ch_mult[i];
            for(int nr = 0;nr<num_res_blocks;nr++) {
            	Resnet3DBlock res3d = new Resnet3DBlock(inc, outc, id, ih, iw, network);
            	downBlock.add(res3d);
            	inc = outc;
            	id = res3d.oDepth;
                ih = res3d.oHeight;
                iw = res3d.oWidth;
            }
            if(i != ch_mult.length - 1) {
            	if(down_sampling_layers.contains(i)) {
            		Downsample3D dwon3d = new Downsample3D(inc, id, ih, iw, 2, network);
            		downBlock.add(dwon3d);
                	id = dwon3d.oDepth;
                    ih = dwon3d.oHeight;
                    iw = dwon3d.oWidth;
            	}else {
            		Downsample2D dwon2d = new Downsample2D(inc, id, ih, iw, network);
            		downBlock.add(dwon2d);
            		ih = dwon2d.oHeight;
                    iw = dwon2d .oWidth;
            	}
            }
        }
        
        // mid
        midBlock = new ArrayList<Layer>();
        Resnet3DBlock mb1 = new Resnet3DBlock(outc, outc, id, ih, iw, network);
        midBlock.add(mb1);
        id = mb1.oDepth;
        ih = mb1.oHeight;
        iw = mb1.oWidth;
        AttentionBlock3D attn = new AttentionBlock3D(outc, id, ih, iw, true, true, network);
        midBlock.add(attn);
        Resnet3DBlock mb2 = new Resnet3DBlock(outc, outc, id, ih, iw, network);
        midBlock.add(mb2);
        id = mb2.oDepth;
        ih = mb2.oHeight;
        iw = mb2.oWidth;
        
        //out
        convNormOut = new GNLayer3D(outc, id, ih, iw, 32, network);
        convAct = new SiLULayer(convNormOut);
        int zc = z_channels;
        if(double_z) {
        	zc = z_channels * 2;
        }
        convOut = new CausalConv3DPlainAR(outc, zc, id, iw, ih, 3, 1, true, network);
        convOut.setUpdater(UpdaterFactory.create(this.network));
        convOut.paramsInit = ParamsInit.silu;
        this.oDepth = convOut.oDepth;
        this.oHeight = convOut.oHeight;
        this.oWidth = convOut.oWidth;
    }

    @Override
    public void init() {
        this.number = this.network.number;
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
        convIn.forward(this.input);

        Tensor x = convIn.getOutput();
       
        for (int i = 0; i < downBlock.size(); i++) {
            Layer layer = downBlock.get(i);
            layer.forward(x);
            x = layer.getOutput();
        }

        for (int i = 0; i < midBlock.size(); i++) {
        	Layer layer = midBlock.get(i);
            layer.forward(x);
            x = layer.getOutput();
        }
        
        convNormOut.forward(x);
        convAct.forward(convNormOut.getOutput());
        convOut.forward(convAct.getOutput());
        this.output = convOut.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        convOut.back(delta);
        convAct.back(convOut.diff);
        convNormOut.back(convAct.diff);
        Tensor d = convNormOut.diff;
        for (int i = midBlock.size() - 1; i >= 0; i--) {
            Layer mid = midBlock.get(i);
            mid.back(d);
            d = mid.diff;
        }
        for (int i = downBlock.size() - 1; i >= 0; i--) {
            Layer down = downBlock.get(i);
            down.back(d);
            d = down.diff;
        }
        convIn.back(d);
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
        for (int i = 0; i < downBlock.size(); i++) {
            downBlock.get(i).update();
        }
        for (int i = 0; i < midBlock.size(); i++) {
            midBlock.get(i).update();
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
}

