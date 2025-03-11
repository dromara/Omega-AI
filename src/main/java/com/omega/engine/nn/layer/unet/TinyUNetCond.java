package com.omega.engine.nn.layer.unet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffusion.TinyTimeEmbeddingLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * UNet_Cond
 * @author Administrator
 *
 */
public class TinyUNetCond extends Layer{
	
	private int[] downChannels;
	
	private int[] midChannels;
	
	private int timeSteps;
	
	private int tEmbDim;
	
	private int groups = 32;
	
	private int headNum;
	
	private int textEmbedDim;
	
	private int maxContextLen;
	
	/**
	 * layers
	 */
	public ConvolutionLayer conv_in1;
	public SiLULayer in_act;
	public ConvolutionLayer conv_in2;
	
	
	public TinyTimeEmbeddingLayer t_embd;
	
	/**
	 * downs
	 */
	public List<UNetDownBlockLayer2> downs;
	
	/**
	 * mids
	 */
	public List<UNetMidBlockLayer2> mids;
	
	/**
	 * ups
	 */
	public List<UNetUpBlockLayer2> ups;
	
	public List<RouteLayer> cats;
	
	/**
	 * fairs
	 */
	private UNetResnetBlockLayer2 resnet;
	
	private ConvolutionLayer conv_out1;
	private SiLULayer act;
	private ConvolutionLayer conv_out2;
	
	private Tensor tDiff;
	
	public TinyUNetCond(int channel,int height,int width,int[] downChannels,int[] midChannels,int timeSteps,
			int tEmbDim,int groups,int headNum,int textEmbedDim,int maxContextLen,Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = channel;
		this.height = height;
		this.width = width;
		this.downChannels = downChannels;
		this.midChannels = midChannels;
		this.timeSteps = timeSteps;
		this.tEmbDim = tEmbDim;
		this.groups = groups;
		this.headNum = headNum;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		initLayers();
	}
	
	public void initLayers() {
		
		conv_in1 = new ConvolutionLayer(channel, downChannels[0], width, height, 3, 3, 1, 1, true, network);
		conv_in1.setUpdater(UpdaterFactory.create(this.network));
		conv_in1.paramsInit = ParamsInit.silu;
		
		in_act = new SiLULayer(conv_in1);
		
		conv_in2 = new ConvolutionLayer(downChannels[0], downChannels[0], width, height, 3, 3, 1, 1, true, network);
		conv_in2.setUpdater(UpdaterFactory.create(this.network));
		conv_in2.paramsInit = ParamsInit.silu;
		
		t_embd = new TinyTimeEmbeddingLayer(timeSteps, tEmbDim, true, network);
		
		int iw = conv_in2.oWidth;
		int ih = conv_in2.oHeight;
		
		
		Stack<Layer> downLayers = new Stack<Layer>();
		
		downLayers.push(conv_in2);
		
		downs = new ArrayList<UNetDownBlockLayer2>();
		mids = new ArrayList<UNetMidBlockLayer2>();
		ups = new ArrayList<UNetUpBlockLayer2>();
		cats = new ArrayList<RouteLayer>();
		
		for(int i = 0;i<downChannels.length - 1;i++) {
			
			UNetDownBlockLayer2 down = new UNetDownBlockLayer2(downChannels[i], downChannels[i+1], ih, iw, tEmbDim, headNum, groups, textEmbedDim, maxContextLen, network);
			downs.add(down);
			iw = down.oWidth;
			ih = down.oHeight;
			if(i < downChannels.length - 2) {
				downLayers.push(down);
			}
		}
		
		int ic = downChannels[downChannels.length - 1];
		
		for(int i = 0;i<midChannels.length;i++) {
			UNetMidBlockLayer2 mid = new UNetMidBlockLayer2(ic, midChannels[i], ih, iw, tEmbDim, headNum, groups, textEmbedDim, maxContextLen, network);
			mids.add(mid);
			ic = midChannels[i];
			iw = mid.oWidth;
			ih = mid.oHeight;
		}

		for(int i = downChannels.length - 1;i>0;i--) {
			if(i == downChannels.length - 1) {
				ic = downChannels[i];
			}else {
				ic = downChannels[i] * 2;
			}
		
			UNetUpBlockLayer2 up = new UNetUpBlockLayer2(ic, downChannels[i - 1], ih, iw, tEmbDim, headNum, groups, textEmbedDim, maxContextLen, network);
			ups.add(up);
			ic = up.oChannel;
			iw = up.oWidth;
			ih = up.oHeight;
			RouteLayer r = new RouteLayer(new Layer[] {up, downLayers.pop()});
			cats.add(r);
		}
		
		resnet = new UNetResnetBlockLayer2(downChannels[0] * 2, downChannels[0], ih, iw, tEmbDim, groups, network);
		
		conv_out1 = new ConvolutionLayer(downChannels[0], downChannels[0], iw, ih, 3, 3, 1, 1, true, this.network);
		conv_out1.setUpdater(UpdaterFactory.create(this.network));
		conv_out1.paramsInit = ParamsInit.silu;
		
		act = new SiLULayer(conv_out1);
		
		conv_out2 = new ConvolutionLayer(downChannels[0], channel, width, height, 3, 3, 1, 1, true, this.network);
		conv_out2.setUpdater(UpdaterFactory.create(this.network));
		conv_out2.paramsInit = ParamsInit.silu;
		
		this.oHeight = ih;
		this.oWidth = iw;
//		System.out.println("activeLayer.preLayer:"+activeLayer.preLayer);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(tDiff == null || tDiff.number != this.number) {
			tDiff = Tensor.createGPUTensor(tDiff, this.number, 1, 1, tEmbDim, true);
		}else {
			tDiff.clearGPU();
		}
		for(int i = 0;i<ups.size();i++) {
			if(ups.get(i).upSampleConv.cache_delta != null) {
				ups.get(i).upSampleConv.cache_delta.clearGPU();
			}
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

	}
	
	public void output(Tensor t,Tensor cond_input) {
		// TODO Auto-generated method stub
		/**
		 * in
		 */
		conv_in1.forward(input);
		in_act.forward(conv_in1.getOutput());
		conv_in2.forward(in_act.getOutput());
		
		/**
		 * t_embd
		 */
		t_embd.forward(t);
		
		Tensor x = conv_in2.getOutput();
		Tensor tembd = t_embd.getOutput();

		/**
		 * down
		 */
		x.showDM("x1");
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).forward(x, tembd, cond_input);
			x = downs.get(i).getOutput();
//			x.showDM(i+":");
		}
		x.showDM("down");
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
//			x.showDM("mids-input");
			mids.get(i).forward(x, tembd, cond_input);
			x = mids.get(i).getOutput();
		}
		x.showDM("mid");
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).forward(x, tembd, cond_input);
			cats.get(i).forward(ups.get(i).getOutput());
			x = cats.get(i).getOutput();
		}
		x.showDM("up");
		/**
		 * out
		 */
		resnet.forward(x, t);
		conv_out1.forward(resnet.getOutput());
		act.forward(conv_out1.getOutput());
		conv_out2.forward(act.getOutput());
		
		this.output = conv_out2.getOutput();
	}


	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		/**
		 * out backward
		 */
//		System.err.println("in----------back");
//		delta.showShape();
		conv_out2.back(delta);
//		conv_out.diff.showShape();
		act.back(conv_out2.diff);
		conv_out1.back(act.diff);
//		act.diff.showDM();
		resnet.back(conv_out1.diff, tDiff);
		
		Tensor d = resnet.diff;
		
//		d.showDM("norm.diff");
		
		/**
		 * ups backward
		 */
		for(int i = ups.size() - 1;i>=0;i--) {
			cats.get(i).back(d);
			ups.get(i).back(ups.get(i).cache_delta, tDiff);
			d = ups.get(i).diff;
		}
		
//		d.showDM("ups.diff");

		/**
		 * mids backward
		 */
		for(int i = mids.size() - 1;i>=0;i--) {
			mids.get(i).back(d, tDiff);
			d = mids.get(i).diff;
		}
		
//		d.showDM("mids.diff");
		
//		d.showDMByOffsetRed(0, 100, "mids.diff");
		
		/**
		 * downs backward
		 */
		for(int i = downs.size() - 1;i>=0;i--) {
			downs.get(i).back(d, tDiff);
			d = downs.get(i).diff;
		}
		
//		d.showDM("downs.diff");
//		tDiff.showDM("tDiff");
		t_embd.back(tDiff);
		
		tDiff.clearGPU();
//		d.showDMByOffsetRed(0, 100, "last Diff");

		conv_in2.back(d);
//		conv_in2.delta.showDM("downs.diff");
		in_act.back(conv_in2.diff);
		conv_in1.back(in_act.diff);
		
		this.diff = conv_in1.diff;
//		System.err.println("out----------back");
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
	public void backTemp() {
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
	
	public void forward(Tensor input,Tensor t,Tensor context) {
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
		this.output(t, context);
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
	public void update() {
		// TODO Auto-generated method stub
		/**
		 * in
		 */
		conv_in1.update();
		conv_in2.update();
		
		/**
		 * t_embd
		 */
		t_embd.update();
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).update();
		}
		
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).update();
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).update();
		}
		
		/**
		 * out
		 */
		resnet.update();
		conv_out1.update();
		conv_out2.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.unet_down;
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
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		/**
		 * in
		 */
		conv_in1.accGrad(scale);
		conv_in2.accGrad(scale);
		
		/**
		 * t_embd
		 */
		t_embd.accGrad(scale);
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).accGrad(scale);
		}
		
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).accGrad(scale);
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).accGrad(scale);
		}
		
		/**
		 * out
		 */
		resnet.accGrad(scale);
		conv_out1.accGrad(scale);
		conv_out2.accGrad(scale);
	}
	
	public static void main(String[] args) {
		
//		int N = 6;
//		int C = 8;
//		int H = 4096;
//		int W = 4096;
//		System.err.println(N * C * H * W * (long)4);
//		
//		CUDAModules.initContext();
//		
//		Tensor x = new Tensor(N, C, H, W, true);
		
		Transformer tf = new Transformer();
		tf.updater = UpdaterType.adamw;
		tf.CUDNN = true;
		tf.learnRate = 0.001f;
		tf.RUN_MODEL = RunModel.TRAIN;
		
		int N = 2;
		int z_channels = 4;
		int H = 8;
		int W = 8;
		int groups = 16;
		int headNum = 16;
		
		int timeSteps = 1000;
		int tEmbDim = 512;
		
		int[] downChannels = new int[] {64, 128, 256};
		int[] midChannels = new int[] {256};
		
		int textEmbedDim = 512;
		int maxContextLen = 64;
		
		TinyUNetCond unet = new TinyUNetCond(z_channels, H, W, downChannels, midChannels, timeSteps, tEmbDim, groups, headNum, textEmbedDim, maxContextLen, tf);
		
		int dataLen = N * z_channels * H * W;
		
		Tensor im = new Tensor(N, z_channels, H, W, MatrixUtils.order(dataLen, 0.01f, 0.01f), true);
		
		Tensor t = new Tensor(N, 1, 1, 1, new float[] {10, 214}, true);
		
		int textLen = N * maxContextLen * textEmbedDim;
		
		Tensor context = new Tensor(N * maxContextLen, 1, 1, textEmbedDim, MatrixUtils.order(textLen, 0.01f, 0.01f), true);
		
		String weight = "H:\\model\\unet_cond_tiny.json";
		loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), unet, true);
		
		tf.number = N;
		
		unet.forward(im, t, context);
		
		unet.getOutput().showDM();
		
		unet.getOutput().showShape();
		
		Tensor delta = new Tensor(N, z_channels, H, W, MatrixUtils.order(dataLen, 0.01f, 0.01f), true);
		
		for(int r = 0;r<10;r++) {
//			delta.showDMByOffsetRed(0, 100, "delta");
			tf.train_time++;
			unet.forward(im, t, context);
			unet.getOutput().showDMByOffsetRed(0, 10, "out");
//			unet.back(delta);
//			unet.diff.showDMByOffsetRed(0, 10, "diff");
//			System.err.println("dddddd=>"+unet.downs.get(0).resnet1.norm1.diffGamma);
//			unet.update();
		}
		
		
		
		
		
	}
	
	public static void loadWeight(Map<String, Object> weightMap, TinyUNetCond unet, boolean showLayers) {
		if(showLayers) {
			for(String key:weightMap.keySet()) {
				System.out.println(key);
			}
		}
		
		int channels = 2;
		
		/**
		 * conv_in
		 */
		ClipModelUtils.loadData(unet.conv_in1.weight, weightMap, "init_conv.0.weight");
		ClipModelUtils.loadData(unet.conv_in1.bias, weightMap, "init_conv.0.bias");
		ClipModelUtils.loadData(unet.conv_in2.weight, weightMap, "init_conv.2.weight");
		ClipModelUtils.loadData(unet.conv_in2.bias, weightMap, "init_conv.2.bias");
		
		/**
		 * t_proj
		 */
		ClipModelUtils.loadData(unet.t_embd.linear1.weight, weightMap, "time_mlp.0.weight");
		ClipModelUtils.loadData(unet.t_embd.linear1.bias, weightMap, "time_mlp.0.bias");
		ClipModelUtils.loadData(unet.t_embd.linear2.weight, weightMap, "time_mlp.2.weight");
		ClipModelUtils.loadData(unet.t_embd.linear2.bias, weightMap, "time_mlp.2.bias");
		
		/**
		 * downs
		 */
		for(int i = 0;i<channels;i++) {
			
			unet.downs.get(i).resnet1.norm1.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnet1.norm1.gamma, weightMap, 1, "down"+i+".resnet1.norm1.weight");
			unet.downs.get(i).resnet1.norm1.beta = ClipModelUtils.loadData(unet.downs.get(i).resnet1.norm1.beta, weightMap, 1, "down"+i+".resnet1.norm1.bias");
			unet.downs.get(i).resnet1.norm2.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnet1.norm2.gamma, weightMap, 1, "down"+i+".resnet1.norm2.weight");
			unet.downs.get(i).resnet1.norm2.beta = ClipModelUtils.loadData(unet.downs.get(i).resnet1.norm2.beta, weightMap, 1, "down"+i+".resnet1.norm2.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.conv1.weight, weightMap, "down"+i+".resnet1.conv1.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.conv1.bias, weightMap, "down"+i+".resnet1.conv1.bias");
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.conv2.weight, weightMap, "down"+i+".resnet1.conv2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.conv2.bias, weightMap, "down"+i+".resnet1.conv2.bias");
			
			unet.downs.get(i).resnet1.residual.weight = ClipModelUtils.loadData(unet.downs.get(i).resnet1.residual.weight, weightMap, 4, "down"+i+".resnet1.residual_conv.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.residual.bias, weightMap, "down"+i+".resnet1.residual_conv.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.temb.linear.weight, weightMap, "down"+i+".resnet1.time_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet1.temb.linear.bias, weightMap, "down"+i+".resnet1.time_proj.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.qLinerLayer.weight, weightMap, "down"+i+".transformer1.transformer.attn_self.query.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.kLinerLayer.weight, weightMap, "down"+i+".transformer1.transformer.attn_self.key.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.vLinerLayer.weight, weightMap, "down"+i+".transformer1.transformer.attn_self.value.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.oLinerLayer.weight, weightMap, "down"+i+".transformer1.transformer.attn_self.out_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.oLinerLayer.bias, weightMap, "down"+i+".transformer1.transformer.attn_self.out_proj.bias");
			
			unet.downs.get(i).st1.transformer.attn.norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.norm.gamma, weightMap, 1, "down"+i+".transformer1.transformer.norm1.weight");
			unet.downs.get(i).st1.transformer.attn.norm.beta = ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.attn.norm.beta, weightMap, 1, "down"+i+".transformer1.transformer.norm1.bias");
			unet.downs.get(i).st1.transformer.affn.norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.norm.gamma, weightMap, 1, "down"+i+".transformer1.transformer.norm2.weight");
			unet.downs.get(i).st1.transformer.affn.norm.beta = ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.norm.beta, weightMap, 1, "down"+i+".transformer1.transformer.norm2.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.linear1.weight, weightMap, "down"+i+".transformer1.transformer.ffn1.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.linear1.bias, weightMap, "down"+i+".transformer1.transformer.ffn1.0.bias");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.linear2.weight, weightMap, "down"+i+".transformer1.transformer.ffn1.2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st1.transformer.affn.linear2.bias, weightMap, "down"+i+".transformer1.transformer.ffn1.2.bias");
			
			/**
			 * resnet2
			 */
			unet.downs.get(i).resnet2.norm1.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnet2.norm1.gamma, weightMap, 1, "down"+i+".resnet2.norm1.weight");
			unet.downs.get(i).resnet2.norm1.beta = ClipModelUtils.loadData(unet.downs.get(i).resnet2.norm1.beta, weightMap, 1, "down"+i+".resnet2.norm1.bias");
			unet.downs.get(i).resnet2.norm2.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnet2.norm2.gamma, weightMap, 1, "down"+i+".resnet2.norm2.weight");
			unet.downs.get(i).resnet2.norm2.beta = ClipModelUtils.loadData(unet.downs.get(i).resnet2.norm2.beta, weightMap, 1, "down"+i+".resnet2.norm2.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.conv1.weight, weightMap, "down"+i+".resnet2.conv1.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.conv1.bias, weightMap, "down"+i+".resnet2.conv1.bias");
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.conv2.weight, weightMap, "down"+i+".resnet2.conv2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.conv2.bias, weightMap, "down"+i+".resnet2.conv2.bias");
			
//			unet.downs.get(i).resnet2.residual.weight = ClipModelUtils.loadData(unet.downs.get(i).resnet2.residual.weight, weightMap, 4, "down"+i+".resnet2.residual_conv.weight");
//			ClipModelUtils.loadData(unet.downs.get(i).resnet2.residual.bias, weightMap, "down"+i+".resnet2.residual_conv.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.temb.linear.weight, weightMap, "down"+i+".resnet2.time_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnet2.temb.linear.bias, weightMap, "down"+i+".resnet2.time_proj.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.qLinerLayer.weight, weightMap, "down"+i+".transformer2.transformer.attn_cross.query.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.kLinerLayer.weight, weightMap, "down"+i+".transformer2.transformer.attn_cross.key.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.vLinerLayer.weight, weightMap, "down"+i+".transformer2.transformer.attn_cross.value.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.oLinerLayer.weight, weightMap, "down"+i+".transformer2.transformer.attn_cross.out_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.oLinerLayer.bias, weightMap, "down"+i+".transformer2.transformer.attn_cross.out_proj.bias");
			
			unet.downs.get(i).st2.transformer.crossAttn.norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.norm.gamma, weightMap, 1, "down"+i+".transformer2.transformer.norm3.weight");
			unet.downs.get(i).st2.transformer.crossAttn.norm.beta = ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.crossAttn.norm.beta, weightMap, 1, "down"+i+".transformer2.transformer.norm3.bias");
			unet.downs.get(i).st2.transformer.caffn.norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.norm.gamma, weightMap, 1, "down"+i+".transformer2.transformer.norm4.weight");
			unet.downs.get(i).st2.transformer.caffn.norm.beta = ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.norm.beta, weightMap, 1, "down"+i+".transformer2.transformer.norm4.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.linear1.weight, weightMap, "down"+i+".transformer2.transformer.ffn2.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.linear1.bias, weightMap, "down"+i+".transformer2.transformer.ffn2.0.bias");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.linear2.weight, weightMap, "down"+i+".transformer2.transformer.ffn2.2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.transformer.caffn.linear2.bias, weightMap, "down"+i+".transformer2.transformer.ffn2.2.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).st2.contextProj.weight, weightMap, "down"+i+".transformer2.context_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).st2.contextProj.bias, weightMap, "down"+i+".transformer2.context_proj.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.weight, weightMap, "down"+i+".downsample.weight");
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.bias, weightMap, "down"+i+".downsample.bias");
			
		}
		
		/**
		 * mids
		 */
		for(int i = 0;i < 1;i++) {
			unet.mids.get(i).resnet1.norm1.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet1.norm1.gamma, weightMap, 1, "middle_block.resnet1.norm1.weight");
			unet.mids.get(i).resnet1.norm1.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet1.norm1.beta, weightMap, 1, "middle_block.resnet1.norm1.bias");
			unet.mids.get(i).resnet1.norm2.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet1.norm2.gamma, weightMap, 1, "middle_block.resnet1.norm2.weight");
			unet.mids.get(i).resnet1.norm2.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet1.norm2.beta, weightMap, 1, "middle_block.resnet1.norm2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).resnet1.conv1.weight, weightMap, "middle_block.resnet1.conv1.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet1.conv1.bias, weightMap, "middle_block.resnet1.conv1.bias");
			ClipModelUtils.loadData(unet.mids.get(i).resnet1.conv2.weight, weightMap, "middle_block.resnet1.conv2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet1.conv2.bias, weightMap, "middle_block.resnet1.conv2.bias");

			ClipModelUtils.loadData(unet.mids.get(i).resnet1.temb.linear.weight, weightMap, "middle_block.resnet1.time_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet1.temb.linear.bias, weightMap, "middle_block.resnet1.time_proj.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.qLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_self.query.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.kLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_self.key.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.vLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_self.value.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.oLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_self.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.oLinerLayer.bias, weightMap, "middle_block.attn1.transformer.attn_self.out_proj.bias");
			
			unet.mids.get(i).st1.transformer.attn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.norm.gamma, weightMap, 1, "middle_block.attn1.transformer.norm1.weight");
			unet.mids.get(i).st1.transformer.attn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.attn.norm.beta, weightMap, 1, "middle_block.attn1.transformer.norm1.bias");
			unet.mids.get(i).st1.transformer.affn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.norm.gamma, weightMap, 1, "middle_block.attn1.transformer.norm2.weight");
			unet.mids.get(i).st1.transformer.affn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.norm.beta, weightMap, 1, "middle_block.attn1.transformer.norm2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.linear1.weight, weightMap, "middle_block.attn1.transformer.ffn1.0.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.linear1.bias, weightMap, "middle_block.attn1.transformer.ffn1.0.bias");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.linear2.weight, weightMap, "middle_block.attn1.transformer.ffn1.2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.affn.linear2.bias, weightMap, "middle_block.attn1.transformer.ffn1.2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.qLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_cross.query.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.kLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_cross.key.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.vLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_cross.value.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.oLinerLayer.weight, weightMap, "middle_block.attn1.transformer.attn_cross.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.oLinerLayer.bias, weightMap, "middle_block.attn1.transformer.attn_cross.out_proj.bias");
			
			unet.mids.get(i).st1.transformer.crossAttn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.norm.gamma, weightMap, 1, "middle_block.attn1.transformer.norm3.weight");
			unet.mids.get(i).st1.transformer.crossAttn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.crossAttn.norm.beta, weightMap, 1, "middle_block.attn1.transformer.norm3.bias");
			unet.mids.get(i).st1.transformer.caffn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.norm.gamma, weightMap, 1, "middle_block.attn1.transformer.norm4.weight");
			unet.mids.get(i).st1.transformer.caffn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.norm.beta, weightMap, 1, "middle_block.attn1.transformer.norm4.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.linear1.weight, weightMap, "middle_block.attn1.transformer.ffn2.0.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.linear1.bias, weightMap, "middle_block.attn1.transformer.ffn2.0.bias");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.linear2.weight, weightMap, "middle_block.attn1.transformer.ffn2.2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.transformer.caffn.linear2.bias, weightMap, "middle_block.attn1.transformer.ffn2.2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st1.contextProj.weight, weightMap, "middle_block.attn1.context_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st1.contextProj.bias, weightMap, "middle_block.attn1.context_proj.bias");
			
			/**
			 * resnet2
			 */
			unet.mids.get(i).resnet2.norm1.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet2.norm1.gamma, weightMap, 1, "middle_block.resnet2.norm1.weight");
			unet.mids.get(i).resnet2.norm1.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet2.norm1.beta, weightMap, 1, "middle_block.resnet2.norm1.bias");
			unet.mids.get(i).resnet2.norm2.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet2.norm2.gamma, weightMap, 1, "middle_block.resnet2.norm2.weight");
			unet.mids.get(i).resnet2.norm2.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet2.norm2.beta, weightMap, 1, "middle_block.resnet2.norm2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).resnet2.conv1.weight, weightMap, "middle_block.resnet2.conv1.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet2.conv1.bias, weightMap, "middle_block.resnet2.conv1.bias");
			ClipModelUtils.loadData(unet.mids.get(i).resnet2.conv2.weight, weightMap, "middle_block.resnet2.conv2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet2.conv2.bias, weightMap, "middle_block.resnet2.conv2.bias");

			ClipModelUtils.loadData(unet.mids.get(i).resnet2.temb.linear.weight, weightMap, "middle_block.resnet2.time_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet2.temb.linear.bias, weightMap, "middle_block.resnet2.time_proj.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.qLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_self.query.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.kLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_self.key.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.vLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_self.value.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.oLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_self.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.oLinerLayer.bias, weightMap, "middle_block.attn2.transformer.attn_self.out_proj.bias");
			
			unet.mids.get(i).st2.transformer.attn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.norm.gamma, weightMap, 1, "middle_block.attn2.transformer.norm1.weight");
			unet.mids.get(i).st2.transformer.attn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.attn.norm.beta, weightMap, 1, "middle_block.attn2.transformer.norm1.bias");
			unet.mids.get(i).st2.transformer.affn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.norm.gamma, weightMap, 1, "middle_block.attn2.transformer.norm2.weight");
			unet.mids.get(i).st2.transformer.affn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.norm.beta, weightMap, 1, "middle_block.attn2.transformer.norm2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.linear1.weight, weightMap, "middle_block.attn2.transformer.ffn1.0.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.linear1.bias, weightMap, "middle_block.attn2.transformer.ffn1.0.bias");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.linear2.weight, weightMap, "middle_block.attn2.transformer.ffn1.2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.affn.linear2.bias, weightMap, "middle_block.attn2.transformer.ffn1.2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.qLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_cross.query.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.kLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_cross.key.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.vLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_cross.value.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.oLinerLayer.weight, weightMap, "middle_block.attn2.transformer.attn_cross.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.oLinerLayer.bias, weightMap, "middle_block.attn2.transformer.attn_cross.out_proj.bias");
			
			unet.mids.get(i).st2.transformer.crossAttn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.norm.gamma, weightMap, 1, "middle_block.attn2.transformer.norm3.weight");
			unet.mids.get(i).st2.transformer.crossAttn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.crossAttn.norm.beta, weightMap, 1, "middle_block.attn2.transformer.norm3.bias");
			unet.mids.get(i).st2.transformer.caffn.norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.norm.gamma, weightMap, 1, "middle_block.attn2.transformer.norm4.weight");
			unet.mids.get(i).st2.transformer.caffn.norm.beta = ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.norm.beta, weightMap, 1, "middle_block.attn2.transformer.norm4.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.linear1.weight, weightMap, "middle_block.attn2.transformer.ffn2.0.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.linear1.bias, weightMap, "middle_block.attn2.transformer.ffn2.0.bias");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.linear2.weight, weightMap, "middle_block.attn2.transformer.ffn2.2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.transformer.caffn.linear2.bias, weightMap, "middle_block.attn2.transformer.ffn2.2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).st2.contextProj.weight, weightMap, "middle_block.attn2.context_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).st2.contextProj.bias, weightMap, "middle_block.attn2.context_proj.bias");
			
			/**
			 * resnet3
			 */
			unet.mids.get(i).resnet3.norm1.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet3.norm1.gamma, weightMap, 1, "middle_block.resnet3.norm1.weight");
			unet.mids.get(i).resnet3.norm1.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet3.norm1.beta, weightMap, 1, "middle_block.resnet3.norm1.bias");
			unet.mids.get(i).resnet3.norm2.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnet3.norm2.gamma, weightMap, 1, "middle_block.resnet3.norm2.weight");
			unet.mids.get(i).resnet3.norm2.beta = ClipModelUtils.loadData(unet.mids.get(i).resnet3.norm2.beta, weightMap, 1, "middle_block.resnet3.norm2.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).resnet3.conv1.weight, weightMap, "middle_block.resnet3.conv1.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet3.conv1.bias, weightMap, "middle_block.resnet3.conv1.bias");
			ClipModelUtils.loadData(unet.mids.get(i).resnet3.conv2.weight, weightMap, "middle_block.resnet3.conv2.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet3.conv2.bias, weightMap, "middle_block.resnet3.conv2.bias");

			ClipModelUtils.loadData(unet.mids.get(i).resnet3.temb.linear.weight, weightMap, "middle_block.resnet3.time_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).resnet3.temb.linear.bias, weightMap, "middle_block.resnet3.time_proj.bias");
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<channels;i++) {
			
			ClipModelUtils.loadData(unet.ups.get(i).upSampleConv.weight, weightMap, "up"+i+".upsample.weight");
			ClipModelUtils.loadData(unet.ups.get(i).upSampleConv.bias, weightMap, "up"+i+".upsample.bias");
			
			unet.ups.get(i).resnet1.norm1.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet1.norm1.gamma, weightMap, 1, "up"+i+".resnet1.norm1.weight");
			unet.ups.get(i).resnet1.norm1.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet1.norm1.beta, weightMap, 1, "up"+i+".resnet1.norm1.bias");
			unet.ups.get(i).resnet1.norm2.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet1.norm2.gamma, weightMap, 1, "up"+i+".resnet1.norm2.weight");
			unet.ups.get(i).resnet1.norm2.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet1.norm2.beta, weightMap, 1, "up"+i+".resnet1.norm2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.conv1.weight, weightMap, "up"+i+".resnet1.conv1.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.conv1.bias, weightMap, "up"+i+".resnet1.conv1.bias");
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.conv2.weight, weightMap, "up"+i+".resnet1.conv2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.conv2.bias, weightMap, "up"+i+".resnet1.conv2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.temb.linear.weight, weightMap, "up"+i+".resnet1.time_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet1.temb.linear.bias, weightMap, "up"+i+".resnet1.time_proj.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.qLinerLayer.weight, weightMap, "up"+i+".transformer1.transformer.attn_self.query.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.kLinerLayer.weight, weightMap, "up"+i+".transformer1.transformer.attn_self.key.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.vLinerLayer.weight, weightMap, "up"+i+".transformer1.transformer.attn_self.value.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.oLinerLayer.weight, weightMap, "up"+i+".transformer1.transformer.attn_self.out_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.oLinerLayer.bias, weightMap, "up"+i+".transformer1.transformer.attn_self.out_proj.bias");
			
			unet.ups.get(i).st1.transformer.attn.norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.norm.gamma, weightMap, 1, "up"+i+".transformer1.transformer.norm1.weight");
			unet.ups.get(i).st1.transformer.attn.norm.beta = ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.attn.norm.beta, weightMap, 1, "up"+i+".transformer1.transformer.norm1.bias");
			unet.ups.get(i).st1.transformer.affn.norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.norm.gamma, weightMap, 1, "up"+i+".transformer1.transformer.norm2.weight");
			unet.ups.get(i).st1.transformer.affn.norm.beta = ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.norm.beta, weightMap, 1, "up"+i+".transformer1.transformer.norm2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.linear1.weight, weightMap, "up"+i+".transformer1.transformer.ffn1.0.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.linear1.bias, weightMap, "up"+i+".transformer1.transformer.ffn1.0.bias");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.linear2.weight, weightMap, "up"+i+".transformer1.transformer.ffn1.2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st1.transformer.affn.linear2.bias, weightMap, "up"+i+".transformer1.transformer.ffn1.2.bias");
			
			/**
			 * resnet2
			 */
			unet.ups.get(i).resnet2.norm1.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet2.norm1.gamma, weightMap, 1, "up"+i+".resnet2.norm1.weight");
			unet.ups.get(i).resnet2.norm1.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet2.norm1.beta, weightMap, 1, "up"+i+".resnet2.norm1.bias");
			unet.ups.get(i).resnet2.norm2.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet2.norm2.gamma, weightMap, 1, "up"+i+".resnet2.norm2.weight");
			unet.ups.get(i).resnet2.norm2.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet2.norm2.beta, weightMap, 1, "up"+i+".resnet2.norm2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.conv1.weight, weightMap, "up"+i+".resnet2.conv1.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.conv1.bias, weightMap, "up"+i+".resnet2.conv1.bias");
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.conv2.weight, weightMap, "up"+i+".resnet2.conv2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.conv2.bias, weightMap, "up"+i+".resnet2.conv2.bias");
			
//			unet.ups.get(i).resnet2.residual.weight = ClipModelUtils.loadData(unet.ups.get(i).resnet2.residual.weight, weightMap, 4, "up"+i+".resnet2.residual_conv.weight");
//			ClipModelUtils.loadData(unet.ups.get(i).resnet2.residual.bias, weightMap, "up"+i+".resnet2.residual_conv.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.temb.linear.weight, weightMap, "up"+i+".resnet2.time_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet2.temb.linear.bias, weightMap, "up"+i+".resnet2.time_proj.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.qLinerLayer.weight, weightMap, "up"+i+".transformer2.transformer.attn_cross.query.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.kLinerLayer.weight, weightMap, "up"+i+".transformer2.transformer.attn_cross.key.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.vLinerLayer.weight, weightMap, "up"+i+".transformer2.transformer.attn_cross.value.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.oLinerLayer.weight, weightMap, "up"+i+".transformer2.transformer.attn_cross.out_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.oLinerLayer.bias, weightMap, "up"+i+".transformer2.transformer.attn_cross.out_proj.bias");
			
			unet.ups.get(i).st2.transformer.crossAttn.norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.norm.gamma, weightMap, 1, "up"+i+".transformer2.transformer.norm3.weight");
			unet.ups.get(i).st2.transformer.crossAttn.norm.beta = ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.crossAttn.norm.beta, weightMap, 1, "up"+i+".transformer2.transformer.norm3.bias");
			unet.ups.get(i).st2.transformer.caffn.norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.norm.gamma, weightMap, 1, "up"+i+".transformer2.transformer.norm4.weight");
			unet.ups.get(i).st2.transformer.caffn.norm.beta = ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.norm.beta, weightMap, 1, "up"+i+".transformer2.transformer.norm4.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.linear1.weight, weightMap, "up"+i+".transformer2.transformer.ffn2.0.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.linear1.bias, weightMap, "up"+i+".transformer2.transformer.ffn2.0.bias");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.linear2.weight, weightMap, "up"+i+".transformer2.transformer.ffn2.2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.transformer.caffn.linear2.bias, weightMap, "up"+i+".transformer2.transformer.ffn2.2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).st2.contextProj.weight, weightMap, "up"+i+".transformer2.context_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).st2.contextProj.bias, weightMap, "up"+i+".transformer2.context_proj.bias");
			
			/**
			 * resnet3
			 */
			unet.ups.get(i).resnet3.norm1.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet3.norm1.gamma, weightMap, 1, "up"+i+".resnet3.norm1.weight");
			unet.ups.get(i).resnet3.norm1.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet3.norm1.beta, weightMap, 1, "up"+i+".resnet3.norm1.bias");
			unet.ups.get(i).resnet3.norm2.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnet3.norm2.gamma, weightMap, 1, "up"+i+".resnet3.norm2.weight");
			unet.ups.get(i).resnet3.norm2.beta = ClipModelUtils.loadData(unet.ups.get(i).resnet3.norm2.beta, weightMap, 1, "up"+i+".resnet3.norm2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).resnet3.conv1.weight, weightMap, "up"+i+".resnet3.conv1.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet3.conv1.bias, weightMap, "up"+i+".resnet3.conv1.bias");
			ClipModelUtils.loadData(unet.ups.get(i).resnet3.conv2.weight, weightMap, "up"+i+".resnet3.conv2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet3.conv2.bias, weightMap, "up"+i+".resnet3.conv2.bias");

			ClipModelUtils.loadData(unet.ups.get(i).resnet3.temb.linear.weight, weightMap, "up"+i+".resnet3.time_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnet3.temb.linear.bias, weightMap, "up"+i+".resnet3.time_proj.bias");
		}
		
		/**
		 * final resnet
		 */
		unet.resnet.norm1.gamma = ClipModelUtils.loadData(unet.resnet.norm1.gamma, weightMap, 1, "final_conv.0.norm1.weight");
		unet.resnet.norm1.beta = ClipModelUtils.loadData(unet.resnet.norm1.beta, weightMap, 1, "final_conv.0.norm1.bias");
		unet.resnet.norm2.gamma = ClipModelUtils.loadData(unet.resnet.norm2.gamma, weightMap, 1, "final_conv.0.norm2.weight");
		unet.resnet.norm2.beta = ClipModelUtils.loadData(unet.resnet.norm2.beta, weightMap, 1, "final_conv.0.norm2.bias");
		
		ClipModelUtils.loadData(unet.resnet.conv1.weight, weightMap, "final_conv.0.conv1.weight");
		ClipModelUtils.loadData(unet.resnet.conv1.bias, weightMap, "final_conv.0.conv1.bias");
		ClipModelUtils.loadData(unet.resnet.conv2.weight, weightMap, "final_conv.0.conv2.weight");
		ClipModelUtils.loadData(unet.resnet.conv2.bias, weightMap, "final_conv.0.conv2.bias");
		
		unet.resnet.residual.weight = ClipModelUtils.loadData(unet.resnet.residual.weight, weightMap, 4, "final_conv.0.residual_conv.weight");
		ClipModelUtils.loadData(unet.resnet.residual.bias, weightMap, "final_conv.0.resnet1.residual_conv.bias");
		
		ClipModelUtils.loadData(unet.resnet.temb.linear.weight, weightMap, "final_conv.0.resnet1.time_proj.weight");
		ClipModelUtils.loadData(unet.resnet.temb.linear.bias, weightMap, "final_conv.0.resnet1.time_proj.bias");
		
		/**
		 * final conv
		 */
		ClipModelUtils.loadData(unet.conv_out1.weight, weightMap, "final_conv.1.weight");
		ClipModelUtils.loadData(unet.conv_out1.bias, weightMap, "final_conv.1.bias");
		
		ClipModelUtils.loadData(unet.conv_out2.weight, weightMap, "final_conv.3.weight");
		ClipModelUtils.loadData(unet.conv_out2.bias, weightMap, "final_conv.3.bias");
	}
	
}
