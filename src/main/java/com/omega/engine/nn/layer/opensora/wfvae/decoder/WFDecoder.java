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
import com.omega.engine.nn.layer.opensora.wfvae.modules.InverseHaarWaveletTransform2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.InverseHaarWaveletTransform3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFCausalConv3D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFConv2D;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

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
    
    public WFDecoder(int channel, int depth, int height, int width, int num_resblocks, int base_channels, int latent_dim, Network network) {
        this.network = network;
        this.num_resblocks = num_resblocks;
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
    	
    	conv_in = new WFCausalConv3D(latent_dim, base_channels * 4, depth, width, height, 3, 1, 1, true, network);
    	
    	mid = new WFDecoderMid(base_channels * 4, base_channels * 4 + energy_flow_hidden_size, conv_in.oDepth, conv_in.oHeight, conv_in.oWidth, network);
    	
    	connect_l2 = new WFDecoderConnect(energy_flow_hidden_size, 24, mid.oDepth, mid.oHeight, mid.oWidth, 1, network);
    	inverse_wavelet_transform_l2 = new InverseHaarWaveletTransform3D(24, oDepth, depth, base_channels, network);
    	
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
//        int N = 2;
//        int C = 32;
//        int F = 17;
//        int H = 32;
//        int W = 32;
//        
//        int OC = 32;
//        
//        float[] data = RandomUtils.order(N * C * F * H * W, 0.01f, 0.01f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);
//        Transformer nn = new Transformer();
//        nn.CUDNN = true;
//        nn.number = N;
//        
//        WFEncoderDown1 block = new WFEncoderDown1(C, OC, F, H, W, nn);
//        
//    	String path = "H:\\model\\Resnet3DBlock.json";
//    	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), block, true);
//        
//    	block.forward(input);
//    	
//    	block.getOutput().showDM();
//    	
//        float[] data2 = RandomUtils.order(N * C * F * H * W, 0.001f, 0.001f);
//        Tensor delta = new Tensor(N, C * F, H, W, data2, true);
//        
//        block.back(delta);
//    	
//        block.diff.showDM();
    }
    
    public static void loadWeight(Map<String, Object> weightMap, WFDecoder block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
//        block.norm1.norm.gamma = ClipModelUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "norm1.weight");
//    	block.norm1.norm.beta = ClipModelUtils.loadData(block.norm1.norm.beta, weightMap, 1, "norm1.bias");
//    	ClipModelUtils.loadData(block.conv1.weight, weightMap, "conv1.conv.weight", 5);
//        ClipModelUtils.loadData(block.conv1.bias, weightMap, "conv1.conv.bias");
//        block.norm2.norm.gamma = ClipModelUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "norm2.weight");
//    	block.norm2.norm.beta = ClipModelUtils.loadData(block.norm2.norm.beta, weightMap, 1, "norm2.bias");
//    	ClipModelUtils.loadData(block.conv2.weight, weightMap, "conv2.conv.weight", 5);
//        ClipModelUtils.loadData(block.conv2.bias, weightMap, "conv2.conv.bias");
    }
    
    @Override
    public void output() {
        // TODO Auto-generated method stub

    	conv_in.forward(input);
    	
    	mid.forward(conv_in.getOutput());
    	
    	Tensor_OP().getByChannel(mid.getOutput(), connect_l2_in, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, base_channels * 4, energy_flow_hidden_size);
    	connect_l2.forward(connect_l2_in);
    	l2_coeffs = connect_l2.getOutput();
    	inverse_wavelet_transform_l2.forward(l2_coeffs);
    	Tensor l2 = inverse_wavelet_transform_l2.getOutput();
    	
    	Tensor_OP().getByChannel(mid.getOutput(), up2_in, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, 0, base_channels * 4);
    	up2.forward(up2_in);
    	
    	Tensor_OP().getByChannel(up2.getOutput(), connect_l1_in, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, base_channels * 4, energy_flow_hidden_size);
    	connect_l1.forward(connect_l1_in);
    	
    	l1_coeffs = connect_l1.getOutput();
    	Tensor_OP().addByChannel(l1_coeffs, l2, new int[] {number, connect_l1.oChannel, 1, connect_l1.oDepth * connect_l1.oHeight * connect_l1.oWidth}, 0);  //l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
    	inverse_wavelet_transform_l1.forward(l1_coeffs);
    	Tensor l1 = inverse_wavelet_transform_l1.getOutput();
    	
    	Tensor_OP().getByChannel(up2.getOutput(), up1_in, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, 0, base_channels * 4);
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
    	
    	inverse_wavelet_transform_out.back(delta);

    	Tensor_OP().getByChannel(inverse_wavelet_transform_out.diff, l1_delta, new int[] {number, conv_out.oChannel, 1, conv_out.oDepth * conv_out.oHeight * conv_out.oWidth}, 0, 3);
    	
    	conv_out.back(inverse_wavelet_transform_out.diff);
    	act.back(conv_out.diff);
    	norm_out.back(act.diff);
    	
    	Tensor d = norm_out.diff;
    	for(int i = 1;i>=0;i--) {
    		blocks.get(i).back(d);
    		d = blocks.get(i).diff;
    	}
    	
    	up1.back(d);
    	Tensor_OP().getByChannel_back(up2.getOutput(), up1.diff, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, 0, base_channels * 4);
    	
    	inverse_wavelet_transform_l1.back(l1_delta);
    	Tensor_OP().add(l1_coeffs_delta, inverse_wavelet_transform_l1.diff, l1_coeffs_delta);
    	Tensor_OP().getByChannel(l1_coeffs_delta, l2_delta, new int[] {number, connect_l1.oChannel, 1, connect_l1.oDepth * connect_l1.oHeight * connect_l1.oWidth}, 0, 3);
    	
    	connect_l1.back(l1_coeffs_delta);
    	Tensor_OP().getByChannel(up2.getOutput(), connect_l1.diff, new int[] {number, up2.oChannel, 1, up2.oDepth * up2.oHeight * up2.oWidth}, base_channels * 4, energy_flow_hidden_size);
    	
    	up2.back(up2.getOutput());
    	
    	Tensor_OP().getByChannel_back(mid.getOutput(), up2.diff, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, 0, base_channels * 4);
    	
    	inverse_wavelet_transform_l2.back(l2_delta);
    	Tensor_OP().add(l2_coeffs_delta, inverse_wavelet_transform_l2.diff, l2_coeffs_delta);
    	connect_l2.back(l2_coeffs_delta);
    	
    	Tensor_OP().getByChannel_back(mid.getOutput(), connect_l2.diff, new int[] {number, mid.oChannel, 1, mid.oDepth * mid.oHeight * mid.oWidth}, base_channels * 4, energy_flow_hidden_size);
    	
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

