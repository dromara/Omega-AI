package com.omega.engine.nn.layer.vae;//package com.omega.engine.nn.layer.vae;
//
//import java.util.ArrayList;
//import java.util.List;
//
//import com.omega.common.data.Tensor;
//import com.omega.engine.nn.layer.ConvolutionLayer;
//import com.omega.engine.nn.layer.Layer;
//import com.omega.engine.nn.layer.LayerType;
//import com.omega.engine.nn.layer.ParamsInit;
//import com.omega.engine.nn.layer.active.SiLULayer;
//import com.omega.engine.nn.layer.normalization.GNLayer;
//import com.omega.engine.nn.network.Network;
//import com.omega.engine.updater.UpdaterFactory;
//
///**
// * resnet block layer
// * @author Administrator
// *
// */
//public class VAEEncoder extends Layer {
//
//	private int numLayers;
//
//	private int groups = 32;
//
//	private float outputScale = 1.0f;
//
//	private int[] blockOutChannels;
//
//	private int layersPerBlock = 2;
//
//	private boolean midBlockAttn = true;
//
//	private ConvolutionLayer convIn;
//
//	private List<VAEDownEncoderBlock> downBlock;
//
//	private VAEMidBlock midBlock;
//
//	private GNLayer convNormOut;
//
//	private SiLULayer convAct;
//
//	private ConvolutionLayer convOut;
//
//	private boolean addAttns = false;
//
//	public VAEEncoder(int channel,int oChannel,int height,int width,int numLayers,int groups,float outputScale,boolean addAttns, Network network) {
//		this.network = network;
//		this.addAttns = addAttns;
//		this.channel = channel;
//		this.oChannel = channel;
//		this.height = height;
//		this.width = width;
//		this.groups = groups;
//		this.outputScale = outputScale;
//		this.numLayers = numLayers;
//
//		initLayers();
//
//	}
//
//	public void initLayers() {
//
//		convIn = new ConvolutionLayer(channel, blockOutChannels[0], width, height, 3, 3, 1, 1, false, this.network);
//		convIn.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
//		convIn.paramsInit = ParamsInit.silu;
//
//		downBlock = new ArrayList<VAEDownEncoderBlock>(blockOutChannels.length);
//
//		int outc = blockOutChannels[0];
//		int ih = convIn.oHeight;
//		int iw = convIn.oWidth;
//		for(int i = 0;i<blockOutChannels.length;i++) {
//			int inc = outc;
//			outc = blockOutChannels[i];
//			boolean addDownsampler = true;
//			if(i == blockOutChannels.length - 1) {
//				addDownsampler = false;
//			}
//			VAEDownEncoderBlock down = new VAEDownEncoderBlock(inc, outc, ih, iw, layersPerBlock, groups, 1.0f, addDownsampler, network);
//			ih = down.oHeight;
//			iw = down.oWidth;
//			downBlock.add(down);
//		}
//
//		// mid
//		midBlock = new VAEMidBlock(outc, ih, iw, 1, groups, outc, 1.0f, addAttns, network);
//
//		//out
//		convNormOut = new GNLayer(groups, midBlock);
//		convAct = new SiLULayer(convNormOut);
//
//		int convOutChannels = 2 * oChannel;
//		convOut = new ConvolutionLayer(outc, convOutChannels, midBlock.oWidth, midBlock.oHeight, 3, 3, 1, 1, false, this.network);
//		convOut.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
//		convOut.paramsInit = ParamsInit.silu;
//
//	}
//
//	@Override
//	public void init() {
//		this.number = this.network.number;
//	}
//
//	@Override
//	public void initBack() {
//
//	}
//
//	@Override
//	public void initParam() {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void output() {
//		// TODO Auto-generated method stub
//
//		convIn.forward(this.input);
//
//		for(int i = 0;i<blockOutChannels.length;i++) {
//
//		}
//
//		this.output = x;
//	}
//
//	@Override
//	public Tensor getOutput() {
//		// TODO Auto-generated method stub
//		return this.output;
//	}
//
//	@Override
//	public void diff() {
//		// TODO Auto-generated method stub
////		System.out.println(index);
//		Tensor dx = delta;
//		for(int i = numLayers - 1;i>=0;i--) {
//			resnets.get(i).back(dx);
//			dx = resnets.get(i).diff;
//			if(addAttns) {
//				attns.get(i).back(dx);
//				dx = attns.get(i).diff;
//			}
//		}
//
//		res0.back(dx);
//
//		this.diff = res0.diff;
//	}
//
//	@Override
//	public void forward() {
//		// TODO Auto-generated method stub
//
//		/**
//		 * 参数初始化
//		 */
//		this.init();
//
//		/**
//		 * 设置输入
//		 */
//		this.setInput();
//
//		/**
//		 * 计算输出
//		 */
//		this.output();
//
//	}
//
//	@Override
//	public void back() {
//		// TODO Auto-generated method stub
//
//		initBack();
//		/**
//		 * 设置梯度
//		 */
//		this.setDelta();
//		/**
//		 * 计算梯度
//		 */
//		this.diff();
//
//	}
//
//	@Override
//	public void update() {
//		// TODO Auto-generated method stub
//
//		res0.update();
//
//		for(int i = 0;i<numLayers;i++) {
//			if(addAttns) {
//				attns.get(i).update();
//			}
//			resnets.get(i).update();
//		}
//
//	}
//
//	@Override
//	public void showDiff() {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public LayerType getLayerType() {
//		// TODO Auto-generated method stub
//		return LayerType.block;
//	}
//
//	@Override
//	public float[][][][] output(float[][][][] input) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public void initCache() {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void forward(Tensor input) {
//		// TODO Auto-generated method stub
//
//		/**
//		 * 参数初始化
//		 */
//		this.init();
//
//		/**
//		 * 设置输入
//		 */
//		this.setInput(input);
//
//		/**
//		 * 计算输出
//		 */
//		this.output();
//
//	}
//
//	@Override
//	public void back(Tensor delta) {
//		// TODO Auto-generated method stub
//
//		initBack();
//		/**
//		 * 设置梯度
//		 */
//		this.setDelta(delta);
//		/**
//		 * 计算梯度
//		 */
//		this.diff();
//
//	}
//
//	@Override
//	public void backTemp() {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void accGrad(float scale) {
//		// TODO Auto-generated method stub
//
//	}
//
//}
