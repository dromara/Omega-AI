package com.omega.engine.nn.layer.opensora.wfvae.encoder;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.opensora.wfvae.modules.HaarWaveletTransform2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.HaarWaveletTransform3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.LNLayer3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFCausalConv3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFConv2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet2DBlock;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * Resnet3DBlock
 *
 * @author Administrator
 */
public class WFEncoder extends Layer {
	
	public int depth;
	public int oDepth;
	
	private int num_resblocks = 2;
	private int base_channels = 128;
	
	public HaarWaveletTransform3D wavelet_transform_in;
	public HaarWaveletTransform2D wavelet_transform_l1;
	public HaarWaveletTransform3D wavelet_transform_l2;
	
	public WFEncoderDown1 down1;
	public WFEncoderDown2 down2;
	public WFEncoderMid mid;
	
	public WFConv2D connect_l1;
	public WFConv2D connect_l2;
	
	public LNLayer3D norm_out;
	private SiLULayer act;
	public WFCausalConv3D conv_out;
	
	private int energy_flow_hidden_size;
    private int latent_dim = 8;
    
    private Tensor l1_coeffs_tmp1;
    private Tensor l1_coeffs_tmp2;
    
    private Tensor l1_coeffs;
    private Tensor l2_coeffs;
    
    private Tensor h1;
    private Tensor h2;
    
	public int getNum_resblocks() {
	    return num_resblocks;
    }

    public WFEncoder(int channel, int depth, int height, int width, int num_resblocks, int base_channels, int energy_flow_hidden_size, int latent_dim, Network network) {
        this.network = network;
        this.num_resblocks = num_resblocks;
        this.energy_flow_hidden_size = energy_flow_hidden_size;
        this.channel = channel;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.base_channels = base_channels;
        this.latent_dim = latent_dim;
        initLayers();
        this.oChannel = conv_out.oChannel;
        this.oDepth = conv_out.oDepth;
        this.oHeight = conv_out.oHeight;
        this.oWidth = conv_out.oWidth;
    }

    public void initLayers() {
    	
    	wavelet_transform_in = new HaarWaveletTransform3D(channel, depth, width, height, network);
    	
    	wavelet_transform_l1 = new HaarWaveletTransform2D(3, wavelet_transform_in.oDepth, wavelet_transform_in.oHeight, wavelet_transform_in.oWidth, network);
    	connect_l1 = new WFConv2D(12, energy_flow_hidden_size, wavelet_transform_in.oDepth, wavelet_transform_l1.oHeight, wavelet_transform_l1.oWidth, 3, 1, 1, network);
    	
    	wavelet_transform_l2 = new HaarWaveletTransform3D(3, wavelet_transform_in.oDepth, wavelet_transform_l1.oWidth, wavelet_transform_l1.oHeight, network);
    	connect_l2 = new WFConv2D(24, energy_flow_hidden_size, wavelet_transform_l2.oDepth, wavelet_transform_l2.oHeight, wavelet_transform_l2.oWidth, 3, 1, 1, network);

    	down1 = new WFEncoderDown1(24, base_channels, wavelet_transform_in.oDepth, wavelet_transform_in.oHeight, wavelet_transform_in.oWidth, num_resblocks, network);
    	
    	down2 = new WFEncoderDown2(base_channels + energy_flow_hidden_size, base_channels * 2, down1.oDepth, down1.oHeight, down1.oWidth, num_resblocks, network);
    	
    	mid = new WFEncoderMid(base_channels * 2 + energy_flow_hidden_size, base_channels * 4, down2.oDepth, down2.oHeight, down2.oWidth, network);
    	
    	norm_out = new LNLayer3D(base_channels * 4, mid.oDepth, mid.oHeight, mid.oWidth, network);
    	
    	act = new SiLULayer(network);
    	
    	conv_out = new WFCausalConv3D(base_channels * 4, latent_dim * 2, norm_out.oDepth, norm_out.oWidth, norm_out.oHeight, 3, 1, 1, true, network);
    	
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if(this.l1_coeffs_tmp1 == null || l1_coeffs_tmp1.number != this.number) {
        	l1_coeffs_tmp1 = Tensor.createGPUTensor(l1_coeffs_tmp1, number, 3 * wavelet_transform_in.oDepth, wavelet_transform_in.oHeight, wavelet_transform_in.oWidth, true);
        	l1_coeffs_tmp2 = Tensor.createGPUTensor(l1_coeffs_tmp2, number, 3 * wavelet_transform_l1.depth, wavelet_transform_l1.oHeight, wavelet_transform_l1.oWidth, true);
        	h1 = Tensor.createGPUTensor(h1, number, (down1.oChannel + connect_l1.oChannel) * down1.oDepth, down1.oHeight, down1.oWidth, true);
        	h2 = Tensor.createGPUTensor(h2, number, (down2.oChannel + connect_l2.oChannel) * down2.oDepth, down2.oHeight, down2.oWidth, true);
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

        
        String inputPath = "D:\\models\\input_encoder.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, C * F, H, W, true);
        ClipModelUtils.loadData(input, datas, "x", 5);
        
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        int num_resblocks = 2;
        int base_channels = 32;
        int latent_dim = 8;
        int energy_flow_hidden_size = 32;
        WFEncoder encoder = new WFEncoder(C, F, H, W, num_resblocks, base_channels, energy_flow_hidden_size, latent_dim, nn);
        
        String weight = "D:\\models\\wf_encoder.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), encoder, true);
        
        encoder.forward(input);
        
        encoder.getOutput().showDM("out");
        
        String deltaPath = "D:\\models\\delta_encoder.json";
        Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, 48, 8, 8, true);
        ClipModelUtils.loadData(delta, datas2, "delta", 5);
        
        Tensor l1_coeffs_delta = new Tensor(N, 12 * 5, 16, 16, true);
        Tensor l2_coeffs_delta = new Tensor(N, 24 * 3, 8, 8, true);
        
        encoder.back(delta, l1_coeffs_delta, l2_coeffs_delta);
        
        encoder.diff.showDM("diff");
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFEncoder block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * encoder down1
         */
        ClipModelUtils.loadData(block.down1.conv.conv.weight, weightMap, "down1.0.weight");
        ClipModelUtils.loadData(block.down1.conv.conv.bias, weightMap, "down1.0.bias");
        for(int i = 0;i<block.num_resblocks;i++) {
        	WFResnet2DBlock b2d = block.down1.resBlocks.get(i);
        	b2d.norm1.norm.gamma = ClipModelUtils.loadData(b2d.norm1.norm.gamma, weightMap, 1, "down1."+(i+1)+".norm1.weight");
        	b2d.norm1.norm.beta = ClipModelUtils.loadData(b2d.norm1.norm.beta, weightMap, 1, "down1."+(i+1)+".norm1.bias");
        	ClipModelUtils.loadData(b2d.conv1.weight, weightMap, "down1."+(i+1)+".conv1.weight");
        	ClipModelUtils.loadData(b2d.conv1.bias, weightMap, "down1."+(i+1)+".conv1.bias");
        	b2d.norm2.norm.gamma = ClipModelUtils.loadData(b2d.norm2.norm.gamma, weightMap, 1, "down1."+(i+1)+".norm2.weight");
        	b2d.norm2.norm.beta = ClipModelUtils.loadData(b2d.norm2.norm.beta, weightMap, 1, "down1."+(i+1)+".norm2.bias");
        	ClipModelUtils.loadData(b2d.conv2.weight, weightMap, "down1."+(i+1)+".conv2.weight");
        	ClipModelUtils.loadData(b2d.conv2.bias, weightMap, "down1."+(i+1)+".conv2.bias");
        }
        ClipModelUtils.loadData(block.down1.downsample.conv.weight, weightMap, "down1.3.conv.weight");
        ClipModelUtils.loadData(block.down1.downsample.conv.bias, weightMap, "down1.3.conv.bias");
        
        /**
         * encoder down2
         */
        ClipModelUtils.loadData(block.down2.conv.conv.weight, weightMap, "down2.0.weight");
        ClipModelUtils.loadData(block.down2.conv.conv.bias, weightMap, "down2.0.bias");
        for(int i = 0;i<block.num_resblocks;i++) {
        	WFResnet3DBlock b3d = block.down2.resBlocks.get(i);
        	b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "down2."+(i+1)+".norm1.weight");
        	b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "down2."+(i+1)+".norm1.bias");
        	ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "down2."+(i+1)+".conv1.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "down2."+(i+1)+".conv1.conv.bias");
        	b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "down2."+(i+1)+".norm2.weight");
        	b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "down2."+(i+1)+".norm2.bias");
        	ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "down2."+(i+1)+".conv2.conv.weight", 5);
        	ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "down2."+(i+1)+".conv2.conv.bias");
        }
        ClipModelUtils.loadData(block.down2.downsample3d.conv.weight, weightMap, "down2.3.conv.conv.weight", 5);
        ClipModelUtils.loadData(block.down2.downsample3d.conv.bias, weightMap, "down2.3.conv.conv.bias");
        
        ClipModelUtils.loadData(block.connect_l1.conv.weight, weightMap, "connect_l1.weight");
        ClipModelUtils.loadData(block.connect_l1.conv.bias, weightMap, "connect_l1.bias");
        ClipModelUtils.loadData(block.connect_l2.conv.weight, weightMap, "connect_l2.weight");
        ClipModelUtils.loadData(block.connect_l2.conv.bias, weightMap, "connect_l2.bias");
        
        block.mid.block1.norm1.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm1.norm.gamma, weightMap, 1, "mid.0.norm1.weight"); 
        block.mid.block1.norm1.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm1.norm.beta, weightMap, 1, "mid.0.norm1.bias");
        ClipModelUtils.loadData(block.mid.block1.conv1.weight, weightMap, "mid.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv1.bias, weightMap, "mid.0.conv1.conv.bias");
        block.mid.block1.norm2.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm2.norm.gamma, weightMap, 1, "mid.0.norm2.weight"); 
        block.mid.block1.norm2.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm2.norm.beta, weightMap, 1, "mid.0.norm2.bias");
        ClipModelUtils.loadData(block.mid.block1.conv2.weight, weightMap, "mid.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv2.bias, weightMap, "mid.0.conv2.conv.bias");
        ClipModelUtils.loadData(block.mid.block1.shortcut.weight, weightMap, "mid.0.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.shortcut.bias, weightMap, "mid.0.nin_shortcut.conv.bias");
        
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
        
        block.norm_out.norm.gamma = ClipModelUtils.loadData(block.norm_out.norm.gamma, weightMap, 1, "norm_out.weight"); 
        block.norm_out.norm.beta = ClipModelUtils.loadData(block.norm_out.norm.beta, weightMap, 1, "norm_out.bias");
        ClipModelUtils.loadData(block.conv_out.weight, weightMap, "conv_out.conv.weight", 5);
        ClipModelUtils.loadData(block.conv_out.bias, weightMap, "conv_out.conv.bias");
        
    }
    
    @Override
    public void output() {
        // TODO Auto-generated method stub

    	wavelet_transform_in.forward(input);

    	int dhw = wavelet_transform_in.oDepth * wavelet_transform_in.oHeight * wavelet_transform_in.oWidth;
    	Tensor_OP().getByChannel(wavelet_transform_in.getOutput(), l1_coeffs_tmp1, new int[] {number, 8 * channel, 1, dhw}, 0, 3);
    	
    	wavelet_transform_l1.forward(l1_coeffs_tmp1);
    	l1_coeffs = wavelet_transform_l1.getOutput();
    	connect_l1.forward(l1_coeffs);

    	int l1_dhw = wavelet_transform_l1.oDepth * wavelet_transform_l1.oHeight * wavelet_transform_l1.oWidth;
    	Tensor_OP().getByChannel(l1_coeffs, l1_coeffs_tmp2, new int[] {number, 4 * channel, 1, l1_dhw}, 0, 3);
    	
    	wavelet_transform_l2.forward(l1_coeffs_tmp2);
    	l2_coeffs = wavelet_transform_l2.getOutput();

    	connect_l2.forward(l2_coeffs);

    	Tensor l1 = connect_l1.getOutput();
    	Tensor l2 = connect_l2.getOutput();

    	down1.forward(wavelet_transform_in.getOutput());
//    	down1.getOutput().showDM("down1");
    	Tensor_OP().cat(down1.getOutput(), l1, h1);

    	down2.forward(h1);
//    	down2.getOutput().showDM("down2");
    	Tensor_OP().cat(down2.getOutput(), l2, h2);
    	
    	mid.forward(h2);
    	
    	norm_out.forward(mid.getOutput());
    	act.forward(norm_out.getOutput());
    	conv_out.forward(act.getOutput());
    	
    	this.output = conv_out.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
//    	
//    	Tensor l1_delta = connect_l1.getOutput();
//    	Tensor l2_delta = connect_l2.getOutput();
//    	
//    	conv_out.back(delta);
//    	norm_out.back(conv_out.diff);
//    	
//    	mid.back(norm_out.diff);
//    	
//    	Tensor_OP().cat_back(mid.diff, down2.getOutput(), l2_delta);
//    	down2.back(down2.getOutput());
//    	Tensor_OP().cat_back(down2.diff, down1.getOutput(), l1_delta);
//    	down1.back(down1.getOutput());
//    	
//    	connect_l2.back(l2_delta);
//    	wavelet_transform_l2.back(connect_l2.diff, l1_coeffs_tmp2);
//    	
//    	
//    	
//    	block1.back(attn.diff);
//    	
//    	this.diff = block1.diff;
    }
    
    public void diff(Tensor l1_coeffs_delta,Tensor l2_coeffs_delta) {
        // TODO Auto-generated method stub
    	Tensor l1_delta = connect_l1.getOutput();
    	Tensor l2_delta = connect_l2.getOutput();

    	conv_out.back(delta);
    	act.back(conv_out.diff);
    	norm_out.back(act.diff);
    	
    	mid.back(norm_out.diff);

    	Tensor_OP().getByChannel(mid.diff, down2.getOutput(), new int[] {number, mid.channel, mid.depth, mid.height * mid.width}, 0);
    	Tensor_OP().getByChannel(mid.diff, l2_delta, new int[] {number, mid.channel, 1, mid.depth * mid.height * mid.width}, down2.oChannel);

    	down2.back(down2.getOutput());

    	Tensor_OP().getByChannel(down2.diff, down1.getOutput(), new int[] {number, down2.channel, 1, down2.depth * down2.height * down2.width}, 0);
    	Tensor_OP().getByChannel(down2.diff, l1_delta, new int[] {number, down2.channel, 1, down2.depth * down2.height * down2.width}, down1.oChannel);
    
    	down1.back(down1.getOutput());
    	
    	connect_l2.back(l2_delta);
    	Tensor_OP().add(connect_l2.diff, l2_coeffs_delta, connect_l2.diff);
    	wavelet_transform_l2.back(connect_l2.diff);
    	
//    	wavelet_transform_l2.diff.showDM("wavelet_transform_l2.diff");

    	connect_l1.back(l1_delta);

    	Tensor_OP().add(connect_l1.diff, l1_coeffs_delta, connect_l1.diff);
    	int l1_dhw = wavelet_transform_l1.oDepth * wavelet_transform_l1.oHeight * wavelet_transform_l1.oWidth;
    	Tensor_OP().addByChannel(connect_l1.diff, wavelet_transform_l2.diff, new int[] {number, 4 * channel, 1, l1_dhw}, 0);
    	
//    	connect_l1.diff.showDM("wavelet_transform_l1-delta");
    	
    	wavelet_transform_l1.back(connect_l1.diff);
//    	wavelet_transform_l1.diff.showDM("wavelet_transform_l1.diff");
    	int dhw = wavelet_transform_in.oDepth * wavelet_transform_in.oHeight * wavelet_transform_in.oWidth;
    	Tensor_OP().addByChannel(down1.diff, wavelet_transform_l1.diff, new int[] {number, 8 * channel, 1, dhw}, 0);
    	
    	wavelet_transform_in.back(down1.diff);

    	this.diff = wavelet_transform_in.diff;
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
    	connect_l1.update();
    	connect_l2.update();
    	down1.update();
    	down2.update();
    	mid.update();
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
    
    public void back(Tensor delta, Tensor l1_coeffs_delta,Tensor l2_coeffs_delta) {
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
    	connect_l1.saveModel(outputStream);
    	connect_l2.saveModel(outputStream);
    	down1.saveModel(outputStream);
    	down2.saveModel(outputStream);
    	mid.saveModel(outputStream);
    	norm_out.saveModel(outputStream);
    	conv_out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        connect_l1.loadModel(inputStream);
    	connect_l2.loadModel(inputStream);
    	down1.loadModel(inputStream);
    	down2.loadModel(inputStream);
    	mid.loadModel(inputStream);
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

