package com.omega.engine.nn.network;

import com.omega.common.data.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.DoubleConvLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.unet.UNetDownLayer;
import com.omega.engine.nn.layer.unet.UNetUPLayer;
import com.omega.engine.updater.UpdaterType;

/**
 * Recurrent Neural Networks
 * @author Administrator
 *
 */
public class UNet extends Network {

	public int inChannel;
	
	public int outChannel;

	private boolean bias = true;
	
	private InputLayer inputLayer;
	
	private DoubleConvLayer inc;
	private UNetDownLayer down1;
	private UNetDownLayer down2;
	private UNetDownLayer down3;
	private UNetDownLayer down4;
	
	private UNetUPLayer up1;
	private UNetUPLayer up2;
	private UNetUPLayer up3;
	private UNetUPLayer up4;
	
	private ConvolutionLayer outc;
	
	private int width;
	
	private int height;
	
	private int factor = 2;
	
	private boolean bilinear = true;
	
	public UNet(LossType lossType,UpdaterType updater,int inChannel,int outChannel,int width,int height,boolean bilinear,boolean bias) {
		this.lossFunction = LossFactory.create(lossType, this);
		this.bilinear = bilinear;
		this.bias = bias;
		this.updater = updater;
		this.width = width;
		this.height = height;
		this.inChannel = inChannel;
		this.outChannel = outChannel;
		initLayers();
	}
	
	public void initLayers() {
		
		this.inputLayer = new InputLayer(inChannel, height, width);
		
		this.inc = new DoubleConvLayer(inChannel, 64, height, width, ActiveType.relu, this);
		this.down1 = new UNetDownLayer(64, 128, inc.oHeight, inc.oWidth, ActiveType.relu, this);
		this.down2 = new UNetDownLayer(128, 256, down1.oHeight, down1.oWidth, ActiveType.relu, this);
		this.down3 = new UNetDownLayer(256, 512, down2.oHeight, down2.oWidth, ActiveType.relu, this);

		if(!bilinear) {
			factor = 1;
		}
		
		this.down4 = new UNetDownLayer(512, 1024 / factor, down3.oHeight, down3.oWidth, ActiveType.relu, this);
		
		this.up1 = new UNetUPLayer(1024, 512 / factor, down4.oHeight, down4.oWidth, bilinear, down3, ActiveType.relu, this);
		this.up2 = new UNetUPLayer(512, 256 / factor, up1.oHeight, up1.oWidth, bilinear, down2, ActiveType.relu, this);
		this.up3 = new UNetUPLayer(256, 128 / factor, up2.oHeight, up2.oWidth, bilinear, down1, ActiveType.relu, this);
		this.up4 = new UNetUPLayer(128, 64, up3.oHeight, up3.oWidth, bilinear, inc, ActiveType.relu, this);
		
		this.outc = new ConvolutionLayer(64, outChannel, up4.oHeight, up4.oWidth, 1, 1, 0, 1, bias, this);
		
		this.addLayer(inputLayer);
		this.addLayer(inc);
		this.addLayer(down1);
		this.addLayer(down2);
		this.addLayer(down3);
		this.addLayer(down4);
		this.addLayer(up1);
		this.addLayer(up2);
		this.addLayer(up3);
		this.addLayer(up4);
		this.addLayer(outc);
		
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		if(layerList.size() <= 0) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		this.setChannel(layerList.get(0).channel);
		this.height = layerList.get(0).height;
		this.width = layerList.get(0).width;
		
		this.oChannel = this.getLastLayer().oChannel;
		this.oHeight = this.getLastLayer().oHeight;
		this.oWidth = this.getLastLayer().oWidth;
		
		if(layerList.get(0).getLayerType() != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}
		
		System.out.println("the network is ready.");
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.UNET;
	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOutput();
	}
	
	@Override
	public Tensor forward(Tensor input) {
//		System.out.println("en_time:"+en_time+",de_time:"+de_time);
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		inputLayer.forward();
		
		this.inc.forward(input);
		
		this.down1.forward(this.inc.getOutput());
		this.down2.forward(this.down1.getOutput());
		this.down3.forward(this.down2.getOutput());
		this.down4.forward(this.down3.getOutput());
		
		this.up1.forward(this.down4.getOutput());
		this.up2.forward(this.up1.getOutput());
		this.up3.forward(this.up2.getOutput());
		this.up4.forward(this.up3.getOutput());
		
		this.outc.forward(this.up4.getOutput());
		
		return this.outc.getOutput();
	}

	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub
//		lossDiff.showDMByNumber(0);
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);
		
		this.outc.back(lossDiff);
		
		this.up4.back(this.outc.diff);
		this.up3.back(this.up4.diff);
		this.up2.back(this.up3.diff);
		this.up1.back(this.up2.diff);
		
		this.down4.back(this.up1.diff);
		this.down3.back(this.down4.diff);
		this.down2.back(this.down3.diff);
		this.down1.back(this.down2.diff);
		
		this.inc.back(this.down1.diff);
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub

		switch (this.getLastLayer().getLayerType()) {
		case softmax:
//			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
//			softmaxLayer.setCurrentLabel(label);
			break;
		case softmax_cross_entropy:
			SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer)this.getLastLayer();
			softmaxWithCrossEntropyLayer.setCurrentLabel(label);
			break;
		default:
			break;
		}
		
		return this.lossFunction.loss(output, label);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		Tensor t = this.lossFunction.diff(output, label);
//		PrintUtils.printImage(t.data);
		return t;
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Tensor loss(Tensor output, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, loss);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, diff);
	}
	
	public Tensor loss(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, igonre);
	}

	public Tensor lossDiff(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, igonre);
	}

	@Override
	public void putParamters() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void putParamterGrads() {
		// TODO Auto-generated method stub
		
	}
	
//	public void saveModel(RandomAccessFile outputStream) throws IOException {
//		decoder.saveModel(outputStream);
//		System.out.println("decoder save success...");
//		fullyLayer.saveModel(outputStream);
//		System.out.println("fullyLayer save success...");
//	}
//	
//	public void loadModel(RandomAccessFile inputStream) throws IOException {
//		decoder.loadModel(inputStream);
//		fullyLayer.loadModel(inputStream);
//	}
	
}
