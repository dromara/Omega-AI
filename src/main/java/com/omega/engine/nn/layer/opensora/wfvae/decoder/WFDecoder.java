package com.omega.engine.nn.layer.opensora.wfvae.decoder;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.opensora.vae.modules.GNLayer3D;
import com.omega.engine.nn.layer.opensora.wfvae.encoder.WFEncoder;
import com.omega.engine.nn.layer.opensora.wfvae.modules.InverseHaarWaveletTransform2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.InverseHaarWaveletTransform3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFCausalConv3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFConv2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * WFDecoder
 *
 * @author Administrator
 */
public class WFDecoder extends Layer {
	
	public int depth;
	public int oDepth;
	
	private int num_resblocks = 2;
	private int base_channels = 128;
	
	public WFCausalConv3D conv_in;
	
	public WFDecoderMid mid;

	public InverseHaarWaveletTransform2D inverse_wavelet_transform_l1;
	public InverseHaarWaveletTransform3D inverse_wavelet_transform_l2;
	
	public WFDecoderUp2 up2;
	public WFDecoderUp1 up1;
	
	public List<WFResnet3DBlock> blocks;
	
	public WFDecoderConnect connect_l1;
	public WFDecoderConnect connect_l2;
	
	public GNLayer3D norm_out;
	private SiLULayer act;
	public WFConv2D conv_out;
	
	public InverseHaarWaveletTransform3D inverse_wavelet_transform_out;
	
	private int energy_flow_hidden_size;
    private int latent_dim = 8;
    
    private Tensor connect_l2_in;
    private Tensor up2_in;
    private Tensor connect_l1_in;
    private Tensor up1_in;
    
    private Tensor l1_coeffs;
    private Tensor l2_coeffs;
    
    public WFDecoder(int channel, int depth, int height, int width, int num_resblocks, int base_channels, int energy_flow_hidden_size, int latent_dim, Network network) {
        this.network = network;
        this.num_resblocks = num_resblocks;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.base_channels = base_channels;
        this.energy_flow_hidden_size = energy_flow_hidden_size;
        this.latent_dim = latent_dim;
        initLayers();
        this.oChannel = conv_out.oChannel;
        this.oDepth = conv_out.oDepth;
        this.oHeight = conv_out.oHeight;
        this.oWidth = conv_out.oWidth;
    }

    public void initLayers() {
    	
    	conv_in = new WFCausalConv3D(latent_dim, base_channels * 4, depth, width, height, 3, 1, 1, true, network);
    	
    	mid = new WFDecoderMid(base_channels * 4, base_channels * 4 + energy_flow_hidden_size, conv_in.oDepth, conv_in.oHeight, conv_in.oWidth, network);
    	
    	connect_l2 = new WFDecoderConnect(energy_flow_hidden_size, 24, mid.oDepth, mid.oHeight, mid.oWidth, 1, network);
    	inverse_wavelet_transform_l2 = new InverseHaarWaveletTransform3D(24, connect_l2.oDepth, connect_l2.oWidth, connect_l2.oHeight, network);
    	
    	up2 = new WFDecoderUp2(base_channels * 4, base_channels * 4 + energy_flow_hidden_size, mid.oDepth, mid.oHeight, mid.oWidth, num_resblocks, network);
    	
    	connect_l1 = new WFDecoderConnect(energy_flow_hidden_size, 12, up2.oDepth, up2.oHeight, up2.oWidth, 1, network);
    	inverse_wavelet_transform_l1 = new InverseHaarWaveletTransform2D(connect_l1.oChannel, connect_l1.oDepth, connect_l1.oHeight, connect_l1.oWidth, network);
    	
    	up1 = new WFDecoderUp1(base_channels * 4, base_channels * 2, up2.oDepth, up2.oHeight, up2.oWidth, num_resblocks, network);
    	
    	blocks = new ArrayList<WFResnet3DBlock>();
    	
    	int id = up1.oDepth;
    	int ih = up1.oHeight;
    	int iw = up1.oWidth;
    	for(int i = 0;i<2;i++) {
    		WFResnet3DBlock b = new WFResnet3DBlock(base_channels * (i == 0 ? 2: 1), base_channels, id, ih, iw, network);
    		blocks.add(b);
    		id = b.oDepth;
    		ih = b.oHeight;
    		iw = b.oWidth;
    	}
    	
    	norm_out = new GNLayer3D(base_channels, id, ih, iw, 32, network);
    	
    	act = new SiLULayer(network);
    	
    	conv_out = new WFConv2D(base_channels, 24, id, ih, iw, 3, 1, 1, network);
    	
    	inverse_wavelet_transform_out = new InverseHaarWaveletTransform3D(24, conv_out.oDepth, conv_out.oHeight, conv_out.oWidth, network);
    	
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if(this.connect_l2_in == null || connect_l2_in.number != this.number) {
        	connect_l2_in = Tensor.createGPUTensor(connect_l2_in, number, energy_flow_hidden_size * mid.oDepth, mid.oHeight, mid.oWidth, true);
        	up2_in = Tensor.createGPUTensor(up2_in, number, base_channels * 4 * mid.oDepth, mid.oHeight, mid.oWidth, true);
        	connect_l1_in = Tensor.createGPUTensor(connect_l1_in, number, energy_flow_hidden_size * up2.oDepth, up2.oHeight, up2.oWidth, true);
        	up1_in = Tensor.createGPUTensor(up1_in, number, base_channels * 4 * up2.oDepth, up2.oHeight, up2.oWidth, true);
        }
    }

    @Override
    public void initBack() {

    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    public static void main(String[] args) {
        int N = 2;
        int C = 3;
        int F = 9;
        int H = 64;
        int W = 64;

        int OC = 8;
        int OF = 3;
        int OH = 8;
        int OW = 8;
        String inputPath = "D:\\models\\input_decoder.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, OC * OF, OH, OW, true);
        ClipModelUtils.loadData(input, datas, "x", 5);
        
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        int num_resblocks = 2;
        int base_channels = 32;
        int latent_dim = 8;
        int energy_flow_hidden_size = 32;
        WFDecoder decoder = new WFDecoder(C, OF, OH, OW, num_resblocks, base_channels, energy_flow_hidden_size, latent_dim, nn);
        
        String weight = "D:\\models\\wf_decoder.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), decoder, true);
        
        decoder.forward(input);
        
        decoder.getOutput().showDM();
        
        String deltaPath = "D:\\models\\delta_decoder.json";
        Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(delta, datas2, "delta", 5);
        
        Tensor l1_coeffs_delta = new Tensor(N, 12 * 5, 16, 16, true);
        Tensor l2_coeffs_delta = new Tensor(N, 24 * 3, 8, 8, true);
        
        decoder.back(delta, l1_coeffs_delta, l2_coeffs_delta);
        
        decoder.diff.showDM("diff");
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFDecoder block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ClipModelUtils.loadData(block.conv_in.weight, weightMap, "conv_in.conv.weight", 5);
        ClipModelUtils.loadData(block.conv_in.bias, weightMap, "conv_in.conv.bias");
        /**
         * mid blocks
         */
        block.mid.block1.norm1.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm1.norm.gamma, weightMap, 1, "mid.0.norm1.weight"); 
        block.mid.block1.norm1.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm1.norm.beta, weightMap, 1, "mid.0.norm1.bias");
        ClipModelUtils.loadData(block.mid.block1.conv1.weight, weightMap, "mid.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv1.bias, weightMap, "mid.0.conv1.conv.bias");
        block.mid.block1.norm2.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm2.norm.gamma, weightMap, 1, "mid.0.norm2.weight"); 
        block.mid.block1.norm2.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm2.norm.beta, weightMap, 1, "mid.0.norm2.bias");
        ClipModelUtils.loadData(block.mid.block1.conv2.weight, weightMap, "mid.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv2.bias, weightMap, "mid.0.conv2.conv.bias");
        
        block.mid.attn.norm.norm.gamma = ClipModelUtils.loadData(block.mid.attn.norm.norm.gamma, weightMap, 1, "mid.1.norm.weight"); 
        block.mid.attn.norm.norm.beta = ClipModelUtils.loadData(block.mid.attn.norm.norm.beta, weightMap, 1, "mid.1.norm.bias");
        ClipModelUtils.loadData(block.mid.attn.qLinerLayer.weight, weightMap, "mid.1.q.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.kLinerLayer.weight, weightMap, "mid.1.k.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.vLinerLayer.weight, weightMap, "mid.1.v.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.oLinerLayer.weight, weightMap, "mid.1.proj_out.conv.weight", 5);
        
        block.mid.block2.norm1.norm.gamma = ClipModelUtils.loadData(block.mid.block2.norm1.norm.gamma, weightMap, 1, "mid.2.norm1.weight"); 
        block.mid.block2.norm1.norm.beta = ClipModelUtils.loadData(block.mid.block2.norm1.norm.beta, weightMap, 1, "mid.2.norm1.bias");
        ClipModelUtils.loadData(block.mid.block2.conv1.weight, weightMap, "mid.2.conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block2.conv1.bias, weightMap, "mid.2.conv1.conv.bias");
        block.mid.block2.norm2.norm.gamma = ClipModelUtils.loadData(block.mid.block2.norm2.norm.gamma, weightMap, 1, "mid.2.norm2.weight"); 
        block.mid.block2.norm2.norm.beta = ClipModelUtils.loadData(block.mid.block2.norm2.norm.beta, weightMap, 1, "mid.2.norm2.bias");
        ClipModelUtils.loadData(block.mid.block2.conv2.weight, weightMap, "mid.2.conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block2.conv2.bias, weightMap, "mid.2.conv2.conv.bias");
        ClipModelUtils.loadData(block.mid.block2.shortcut.weight, weightMap, "mid.2.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block2.shortcut.bias, weightMap, "mid.2.nin_shortcut.conv.bias");
        
        /**
         * up2
         */
        for(int i = 0;i<block.num_resblocks;i++) {
        	WFResnet3DBlock b3d = block.up2.resBlocks.get(i);
        	b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "up2."+i+".norm1.weight");
        	b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "up2."+i+".norm1.bias");
        	ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "up2."+i+".conv1.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "up2."+i+".conv1.conv.bias");
        	b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "up2."+i+".norm2.weight");
        	b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "up2."+i+".norm2.bias");
        	ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "up2."+i+".conv2.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "up2."+i+".conv2.conv.bias");
        	if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "up2."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "up2."+i+".nin_shortcut.conv.bias");
        	}
        }
        ClipModelUtils.loadData(block.up2.up3d.conv.weight, weightMap, "up2.2.conv.conv.weight", 5);
        ClipModelUtils.loadData(block.up2.up3d.conv.bias, weightMap, "up2.2.conv.conv.bias");
        WFResnet3DBlock up2_b = block.up2.block;
        up2_b.norm1.norm.gamma = ClipModelUtils.loadData(up2_b.norm1.norm.gamma, weightMap, 1, "up2.3.norm1.weight");
        up2_b.norm1.norm.beta = ClipModelUtils.loadData(up2_b.norm1.norm.beta, weightMap, 1, "up2.3.norm1.bias");
    	ClipModelUtils.loadData(up2_b.conv1.weight, weightMap, "up2.3.conv1.conv.weight", 5);
    	ClipModelUtils.loadData(up2_b.conv1.bias, weightMap, "up2.3.conv1.conv.bias");
    	up2_b.norm2.norm.gamma = ClipModelUtils.loadData(up2_b.norm2.norm.gamma, weightMap, 1, "up2.3.norm2.weight");
    	up2_b.norm2.norm.beta = ClipModelUtils.loadData(up2_b.norm2.norm.beta, weightMap, 1, "up2.3.norm2.bias");
    	ClipModelUtils.loadData(up2_b.conv2.weight, weightMap, "up2.3.conv2.conv.weight", 5);
    	ClipModelUtils.loadData(up2_b.conv2.bias, weightMap, "up2.3.conv2.conv.bias");
        ClipModelUtils.loadData(up2_b.shortcut.weight, weightMap, "up2.3.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(up2_b.shortcut.bias, weightMap, "up2.3.nin_shortcut.conv.bias");
        
        /**
         * up1
         */
        for(int i = 0;i<block.num_resblocks;i++) {
        	WFResnet3DBlock b3d = block.up1.resBlocks.get(i);
        	b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "up1."+i+".norm1.weight");
        	b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "up1."+i+".norm1.bias");
        	ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "up1."+i+".conv1.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "up1."+i+".conv1.conv.bias");
        	b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "up1."+i+".norm2.weight");
        	b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "up1."+i+".norm2.bias");
        	ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "up1."+i+".conv2.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "up1."+i+".conv2.conv.bias");
        	if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "up1."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "up1."+i+".nin_shortcut.conv.bias");
        	}
        }
        ClipModelUtils.loadData(block.up1.up2d.conv.weight, weightMap, "up1.2.conv.weight");
        ClipModelUtils.loadData(block.up1.up2d.conv.bias, weightMap, "up1.2.conv.bias");
        WFResnet3DBlock up1_b = block.up1.block;
        up1_b.norm1.norm.gamma = ClipModelUtils.loadData(up1_b.norm1.norm.gamma, weightMap, 1, "up1.3.norm1.weight");
        up1_b.norm1.norm.beta = ClipModelUtils.loadData(up1_b.norm1.norm.beta, weightMap, 1, "up1.3.norm1.bias");
    	ClipModelUtils.loadData(up1_b.conv1.weight, weightMap, "up1.3.conv1.conv.weight", 5);
    	ClipModelUtils.loadData(up1_b.conv1.bias, weightMap, "up1.3.conv1.conv.bias");
    	up1_b.norm2.norm.gamma = ClipModelUtils.loadData(up1_b.norm2.norm.gamma, weightMap, 1, "up1.3.norm2.weight");
    	up1_b.norm2.norm.beta = ClipModelUtils.loadData(up1_b.norm2.norm.beta, weightMap, 1, "up1.3.norm2.bias");
    	ClipModelUtils.loadData(up1_b.conv2.weight, weightMap, "up1.3.conv2.conv.weight", 5);
    	ClipModelUtils.loadData(up1_b.conv2.bias, weightMap, "up1.3.conv2.conv.bias");
        
    	/**
    	 * layers
    	 */
    	for(int i = 0;i<2;i++) {
    		WFResnet3DBlock b3d = block.blocks.get(i);
    		b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "layer."+i+".norm1.weight");
        	b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "layer."+i+".norm1.bias");
        	ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "layer."+i+".conv1.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "layer."+i+".conv1.conv.bias");
        	b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "layer."+i+".norm2.weight");
        	b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "layer."+i+".norm2.bias");
        	ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "layer."+i+".conv2.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "layer."+i+".conv2.conv.bias");
        	if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "layer."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "layer."+i+".nin_shortcut.conv.bias");
        	}
    	}
    	
    	/**
    	 * connect_l1
    	 */
    	WFResnet3DBlock b3d = block.connect_l1.resBlocks.get(0);
    	b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "connect_l1.0.norm1.weight");
    	b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "connect_l1.0.norm1.bias");
    	ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "connect_l1.0.conv1.conv.weight", 5);
    	ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "connect_l1.0.conv1.conv.bias");
    	b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "connect_l1.0.norm2.weight");
    	b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "connect_l1.0.norm2.bias");
    	ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "connect_l1.0.conv2.conv.weight", 5);
    	ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "connect_l1.0.conv2.conv.bias");
    	ClipModelUtils.loadData(block.connect_l1.conv.conv.weight, weightMap, "connect_l1.1.weight");
    	ClipModelUtils.loadData(block.connect_l1.conv.conv.bias, weightMap, "connect_l1.1.bias");

    	/**
    	 * connect_l2
    	 */
    	WFResnet3DBlock b3d2 = block.connect_l2.resBlocks.get(0);
    	b3d2.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "connect_l2.0.norm1.weight");
    	b3d2.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "connect_l2.0.norm1.bias");
    	ClipModelUtils.loadData(b3d2.conv1.weight, weightMap, "connect_l2.0.conv1.conv.weight", 5);
    	ClipModelUtils.loadData(b3d2.conv1.bias, weightMap, "connect_l2.0.conv1.conv.bias");
    	b3d2.norm2.norm.gamma = ClipModelUtils.loadData(b3d2.norm2.norm.gamma, weightMap, 1, "connect_l2.0.norm2.weight");
    	b3d2.norm2.norm.beta = ClipModelUtils.loadData(b3d2.norm2.norm.beta, weightMap, 1, "connect_l2.0.norm2.bias");
    	ClipModelUtils.loadData(b3d2.conv2.weight, weightMap, "connect_l2.0.conv2.conv.weight", 5);
    	ClipModelUtils.loadData(b3d2.conv2.bias, weightMap, "connect_l2.0.conv2.conv.bias");
    	ClipModelUtils.loadData(block.connect_l2.conv.conv.weight, weightMap, "connect_l2.1.weight");
    	ClipModelUtils.loadData(block.connect_l2.conv.conv.bias, weightMap, "connect_l2.1.bias");
    	
        block.norm_out.norm.gamma = ClipModelUtils.loadData(block.norm_out.norm.gamma, weightMap, 1, "norm_out.weight");
        block.norm_out.norm.beta = ClipModelUtils.loadData(    block.norm_out.norm.beta, weightMap, 1, "norm_out.bias");
    	ClipModelUtils.loadData(block.conv_out.conv.weight, weightMap, "conv_out.weight");
    	ClipModelUtils.loadData(block.conv_out.conv.bias, weightMap, "conv_out.bias");
        
    }
    
    @Override
    public void output() {
        // TODO Auto-generated method stub
//    	input.showDM("input");
    	conv_in.forward(input);
//    	conv_in.getOutput().showDM("conv_in");
    	mid.forward(conv_in.getOutput());
    	
//    	mid.getOutput().showDM("mid");
    	
    	Tensor_OP().getByChannel(mid.getOutput(), connect_l2_in, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, base_channels * 4);
    	connect_l2.forward(connect_l2_in);
    	l2_coeffs = connect_l2.getOutput();
    	
//    	l2_coeffs.showDM("l2_coeffs");
    	
    	inverse_wavelet_transform_l2.forward(l2_coeffs);
    	Tensor l2 = inverse_wavelet_transform_l2.getOutput();
    	
//    	l2.showDM("l2");
    	
    	Tensor_OP().getByChannel(mid.getOutput(), up2_in, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, 0);
    	up2.forward(up2_in);
    	
    	Tensor_OP().getByChannel(up2.getOutput(), connect_l1_in, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, base_channels * 4);
    	connect_l1.forward(connect_l1_in);
    	l1_coeffs = connect_l1.getOutput();

    	Tensor_OP().addByChannel(l1_coeffs, l2, new int[] {number, connect_l1.oChannel, 1, connect_l1.oDepth * connect_l1.oHeight * connect_l1.oWidth}, 0);  //l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
    	inverse_wavelet_transform_l1.forward(l1_coeffs);
    	Tensor l1 = inverse_wavelet_transform_l1.getOutput();
//    	l1.showDM("l1");
    	Tensor_OP().getByChannel(up2.getOutput(), up1_in, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, 0);
    	up1.forward(up1_in);
    	
    	Tensor x = up1.getOutput();
    	for(int i = 0;i<2;i++) {
    		blocks.get(i).forward(x);
    		x = blocks.get(i).getOutput();
    	}
    	
    	norm_out.forward(x);
    	act.forward(norm_out.getOutput());
    	conv_out.forward(act.getOutput());
    	
    	Tensor_OP().addByChannel(conv_out.getOutput(), l1, new int[] {number, conv_out.oChannel, 1, conv_out.oDepth * conv_out.oHeight * conv_out.oWidth}, 0);  //h[:, :3] = h[:, :3] + l1
    	
    	inverse_wavelet_transform_out.forward(conv_out.getOutput());

    	this.output = inverse_wavelet_transform_out.getOutput();
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
    
    public void diff(Tensor l1_coeffs_delta,Tensor l2_coeffs_delta) {
        // TODO Auto-generated method stub
    	Tensor l1_delta = inverse_wavelet_transform_l1.getOutput();
    	Tensor l2_delta = inverse_wavelet_transform_l2.getOutput();
    	
//    	delta.showDM("delta");
    	
    	inverse_wavelet_transform_out.back(delta);
    	
    	Tensor_OP().getByChannel(inverse_wavelet_transform_out.diff, l1_delta, new int[] {number, conv_out.oChannel, 1, conv_out.oDepth * conv_out.oHeight * conv_out.oWidth}, 0, 3);
    	
    	conv_out.back(inverse_wavelet_transform_out.diff);
//    	conv_out.diff.showDM("conv_out.diff");
    	act.back(conv_out.diff);
    	norm_out.back(act.diff);
    	
    	Tensor d = norm_out.diff;
    	for(int i = 1;i>=0;i--) {
    		blocks.get(i).back(d);
    		d = blocks.get(i).diff;
    	}

    	up1.back(d);
//    	up1.diff.showDM("up1");
    	Tensor_OP().getByChannel_back(up2.getOutput(), up1.diff, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, 0);
    	
    	inverse_wavelet_transform_l1.back(l1_delta);

    	Tensor_OP().add(l1_coeffs_delta, inverse_wavelet_transform_l1.diff, l1_coeffs_delta);
    	Tensor_OP().getByChannel(l1_coeffs_delta, l2_delta, new int[] {number, inverse_wavelet_transform_l1.channel, 1, inverse_wavelet_transform_l1.depth * inverse_wavelet_transform_l1.height * inverse_wavelet_transform_l1.width}, 0, 3);
    	
    	connect_l1.back(l1_coeffs_delta);
//    	connect_l1.diff.showDM("connect_l1.diff");
    	Tensor_OP().getByChannel_back(up2.getOutput(), connect_l1.diff, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, base_channels * 4);
    	
//    	up2.getOutput().showDM("up2_delta");
    	
    	up2.back(up2.getOutput());
    	
//    	up2.diff.showDM("up2_diff");
    	
    	Tensor_OP().getByChannel_back(mid.getOutput(), up2.diff, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, 0, base_channels * 4);
    	
    	inverse_wavelet_transform_l2.back(l2_delta);
    	Tensor_OP().add(l2_coeffs_delta, inverse_wavelet_transform_l2.diff, l2_coeffs_delta);
    	connect_l2.back(l2_coeffs_delta);
    	
    	Tensor_OP().getByChannel_back(mid.getOutput(), connect_l2.diff, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, base_channels * 4, energy_flow_hidden_size);
    	
//    	mid.getOutput().showDM("mid-delta");
    	
    	mid.back(mid.getOutput());
    	
    	conv_in.back(mid.diff);
    	
    	this.diff = conv_in.diff;
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
    	conv_in.update();
    	mid.update();
    	connect_l2.update();
    	up2.update();
    	connect_l1.update();
    	up1.update();
    	for(int i = 0;i<2;i++) {
    		blocks.get(i).update();
    	}
    	norm_out.update();
    	conv_out.update();
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
    
    public void back(Tensor delta,Tensor l1_coeffs_delta,Tensor l2_coeffs_delta) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(l1_coeffs_delta, l2_coeffs_delta);
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
    	mid.saveModel(outputStream);
    	connect_l2.saveModel(outputStream);
    	up2.saveModel(outputStream);
    	connect_l1.saveModel(outputStream);
    	up1.saveModel(outputStream);
    	for(int i = 0;i<2;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	norm_out.saveModel(outputStream);
    	conv_out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv_in.loadModel(inputStream);
    	mid.loadModel(inputStream);
    	connect_l2.loadModel(inputStream);
    	up2.loadModel(inputStream);
    	connect_l1.loadModel(inputStream);
    	up1.loadModel(inputStream);
    	for(int i = 0;i<2;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	norm_out.loadModel(inputStream);
    	conv_out.loadModel(inputStream);
    }

	public Tensor getL1_coeffs() {
		return l1_coeffs;
	}

	public Tensor getL2_coeffs() {
		return l2_coeffs;
	}
}

