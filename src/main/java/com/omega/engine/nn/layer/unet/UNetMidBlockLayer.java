package com.omega.engine.nn.layer.unet;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * UNetMidBlockLayer
 * @author Administrator
 *
 */
public class UNetMidBlockLayer extends Layer{
	
	private boolean crossAttn = false;
	
	private int tEmbDim;
	
	private int contextDim = 0;
	
	private int maxContextLen = 0;
	
	private int numHeads;
	
	private int numLayers = 1;
	
	private int groups = 32;
	
	public List<UNetResnetBlockLayer> resnetFirst;
	
	public List<UNetTEmbLayer> tEmbLayers;
	
	public List<UNetResnetBlockLayer> resnetSecond;
	
	public List<UNetAttentionLayer> attns;
	
	public List<UNetAttentionLayer> crossAttns;
	
	public List<FullyLayer> contextProjs;
	
	public List<ConvolutionLayer> residualInputs;

	private Tensor[] t_out;
	
	private Tensor[] res_out;
	
	private Tensor kvDiff;
	
	private Tensor dt;
	
	public UNetMidBlockLayer(int channel,int oChannel,int height,int width,int tEmbDim,int numHeads,int numLayers,int groups,int contextDim,int maxContextLen,boolean crossAttn,Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		this.tEmbDim = tEmbDim;
		this.contextDim = contextDim;
		this.maxContextLen = maxContextLen;
		this.numHeads = numHeads;
		this.numLayers = numLayers;
		this.groups = groups;
		this.crossAttn = crossAttn;
		initLayers();
	}
	
	public void initLayers() {
		
		resnetFirst = new ArrayList<UNetResnetBlockLayer>();
		resnetSecond = new ArrayList<UNetResnetBlockLayer>();
		
		for(int i = 0;i<numLayers + 1;i++) {
			int ic = oChannel;
			if(i == 0) {
				ic = channel;
			}
			UNetResnetBlockLayer b = new UNetResnetBlockLayer(ic, oChannel, height, width, groups, network);
			resnetFirst.add(b);
		}
		
		if(tEmbDim > 0) {
			tEmbLayers = new ArrayList<UNetTEmbLayer>();
			for(int i = 0;i<numLayers + 1;i++) {
				UNetTEmbLayer t = new UNetTEmbLayer(tEmbDim, oChannel, network);
				tEmbLayers.add(t);
			}
		}
		
		for(int i = 0;i<numLayers + 1;i++) {
			UNetResnetBlockLayer b = new UNetResnetBlockLayer(oChannel, oChannel, height, width, groups, network);
			resnetSecond.add(b);
		}
		
		attns = new ArrayList<UNetAttentionLayer>();
		for(int i = 0;i<numLayers;i++) {
			UNetAttentionLayer a = new UNetAttentionLayer(oChannel, numHeads, oChannel, height, width, groups, false, false, true, network);
			attns.add(a);
		}
		
		if(crossAttn) {
			contextProjs = new ArrayList<FullyLayer>();
			for(int i = 0;i<numLayers;i++) {
				FullyLayer f = new FullyLayer(contextDim, oChannel, true, network);
				contextProjs.add(f);
			}
			crossAttns = new ArrayList<UNetAttentionLayer>();
			for(int i = 0;i<numLayers;i++) {
				UNetAttentionLayer a = new UNetAttentionLayer(oChannel, oChannel, oChannel, numHeads, maxContextLen, oChannel, height, width, groups, false, false, true, network);
				crossAttns.add(a);
			}
		}
		
		residualInputs = new ArrayList<ConvolutionLayer>();
		for(int i = 0;i<numLayers+1;i++) {
			int ic = oChannel;
			if(i == 0) {
				ic = channel;
			}
			ConvolutionLayer c = new ConvolutionLayer(ic, oChannel, width, height, 1, 1, 0, 1, false, network);
			c.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
			c.paramsInit = ParamsInit.silu;
			residualInputs.add(c);
		}
		
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(res_out == null || res_out[0].number != this.number) {
			if(res_out == null) {
				res_out = new Tensor[numLayers + 1];
			}
			for(int i = 0;i<numLayers + 1;i++) {
				res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
			}
		}
		
		if(tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
			if(t_out == null) {
				t_out = new Tensor[numLayers + 1];
			}
			for(int i = 0;i<numLayers + 1;i++) {
				t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
			}
		}
		
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(res_out == null || res_out[0].number != this.number) {
			if(res_out == null) {
				res_out = new Tensor[numLayers + 1];
			}
			for(int i = 0;i<numLayers + 1;i++) {
				res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
			}
		}
		
		if(tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
			if(t_out == null) {
				t_out = new Tensor[numLayers + 1];
			}
			for(int i = 0;i<numLayers + 1;i++) {
				t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
			}
		}
		
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(crossAttn && (kvDiff == null || kvDiff.number * maxContextLen != this.number * maxContextLen)) {
			kvDiff = Tensor.createGPUTensor(kvDiff, this.number * maxContextLen, 1, 1, oChannel, true);
		}
		if(dt == null || dt.number != this.number) {
			dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		Tensor x = input;
		
		resnetFirst.get(0).forward(x);
		resnetSecond.get(0).forward(resnetFirst.get(0).getOutput());
		residualInputs.get(0).forward(x);
		
		Tensor_OP().add(resnetSecond.get(0).getOutput(), residualInputs.get(0).getOutput(), res_out[0]);
		
		x = res_out[0];
		
		for(int i = 0;i<numLayers;i++) {
			
			attns.get(i).forward(x);
			x = attns.get(i).getOutput();
			
			resnetFirst.get(i+1).forward(x);
			resnetSecond.get(i+1).forward(resnetFirst.get(i+1).getOutput());
			residualInputs.get(i+1).forward(x);
			
			Tensor_OP().add(resnetSecond.get(i+1).getOutput(), residualInputs.get(i+1).getOutput(), res_out[i+1]);
			
			x = res_out[i+1];

		}
		
		this.output = x;
	}
	
	public void output(Tensor tembd) {
		// TODO Auto-generated method stub
		
		Tensor x = input;
		
		resnetFirst.get(0).forward(x);
		
		tEmbLayers.get(0).forward(tembd);
		
		Tensor r1 = resnetFirst.get(0).getOutput();
		
		Tensor_OP().add(r1, tEmbLayers.get(0).getOutput(), t_out[0], r1.height * r1.width);

		resnetSecond.get(0).forward(t_out[0]);
		
		residualInputs.get(0).forward(x);
		
		Tensor_OP().add(resnetSecond.get(0).getOutput(), residualInputs.get(0).getOutput(), res_out[0]);
		
		x = res_out[0];
		
		for(int i = 0;i<numLayers;i++) {
			
			attns.get(i).forward(x);
			x = attns.get(i).getOutput();
//			x.showDM("mids0["+i+"]");

			resnetFirst.get(i+1).forward(x);
			
			tEmbLayers.get(i+1).forward(tembd);
			
			Tensor r = resnetFirst.get(i+1).getOutput();
			
			Tensor_OP().add(r, tEmbLayers.get(i+1).getOutput(), t_out[i+1], r.height * r.width);
			
			resnetSecond.get(i+1).forward(t_out[i+1]);
			
			residualInputs.get(i+1).forward(x);
			
			Tensor_OP().add(resnetSecond.get(i+1).getOutput(), residualInputs.get(i+1).getOutput(), res_out[i+1]);
			
			x = res_out[i+1];
			
//			x.showDM("mids1["+i+"]");
		}
		
		this.output = x;
	}
	
	public void output(Tensor tembd,Tensor context) {
		// TODO Auto-generated method stub
		
		Tensor x = input;
		
		resnetFirst.get(0).forward(x);
		
		tEmbLayers.get(0).forward(tembd);
		
		Tensor r1 = resnetFirst.get(0).getOutput();
		
		Tensor_OP().add(r1, tEmbLayers.get(0).getOutput(), t_out[0], r1.height * r1.width);

		resnetSecond.get(0).forward(t_out[0]);
		
		residualInputs.get(0).forward(x);
		
		Tensor_OP().add(resnetSecond.get(0).getOutput(), residualInputs.get(0).getOutput(), res_out[0]);
		
		x = res_out[0];
		
		for(int i = 0;i<numLayers;i++) {
			
			attns.get(i).forward(x);
			x = attns.get(i).getOutput();
//			x.showDM("mids0["+i+"]");
			if(crossAttn) {
				contextProjs.get(i).forward(context);
				Tensor cp = contextProjs.get(i).getOutput();
				crossAttns.get(i).forward(x, cp, cp);
				x = crossAttns.get(i).getOutput();
			}
			
			resnetFirst.get(i+1).forward(x);
			
			tEmbLayers.get(i+1).forward(tembd);
			
			Tensor r = resnetFirst.get(i+1).getOutput();
			
			Tensor_OP().add(r, tEmbLayers.get(i+1).getOutput(), t_out[i+1], r.height * r.width);
			
			resnetSecond.get(i+1).forward(t_out[i+1]);
			
			residualInputs.get(i+1).forward(x);
			
			Tensor_OP().add(resnetSecond.get(i+1).getOutput(), residualInputs.get(i+1).getOutput(), res_out[i+1]);
			
			x = res_out[i+1];
			
//			x.showDM("mids1["+i+"]");
		}
		
		this.output = x;
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
	
	public void diff(Tensor tDiff) {
		// TODO Auto-generated method stub
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		
		Tensor d = delta;
//		d.showDM("mout");
		for(int i = numLayers - 1;i>=0;i--) {
			
			residualInputs.get(i+1).back(d);
			resnetSecond.get(i+1).back(d);
			
			d = resnetSecond.get(i+1).diff;
			
			if(tEmbDim > 0) {
				dt.clearGPU();
				Tensor_OP().sum(d, dt, 2);
				tEmbLayers.get(i+1).back(dt);
				Tensor_OP().add(tDiff, tEmbLayers.get(i+1).diff, tDiff);
			}
			
			resnetFirst.get(i+1).back(d);
			
			d = resnetFirst.get(i+1).diff;
			
			Tensor_OP().add(d, residualInputs.get(i+1).diff, d);
			
//			d.showDM("mids.crossDelta");
			
			if(crossAttn) {
				crossAttns.get(i).back(d, kvDiff);
				contextProjs.get(i).back(kvDiff);
				d = crossAttns.get(i).diff;
//				d.showDM("crossAttns");
			}
			
			attns.get(i).back(d);
			d = attns.get(i).diff;
//			d.showDM("mids.attn");
		}
		
		residualInputs.get(0).back(d);
		resnetSecond.get(0).back(d);
		
		d = resnetSecond.get(0).diff;
		
		if(tEmbDim > 0) {
			Tensor_OP().sum(d, dt, 2);
			tEmbLayers.get(0).back(dt);
			Tensor_OP().add(tDiff, tEmbLayers.get(0).diff, tDiff);
		}
		
		resnetFirst.get(0).back(d);
		
		d = resnetFirst.get(0).diff;
		
		Tensor_OP().add(d, residualInputs.get(0).diff, d);
		
		this.diff = d;

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
	
	public void forward(Tensor input,Tensor tembd) {
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
		this.output(tembd);
	}
	
	public void forward(Tensor input,Tensor tembd,Tensor context) {
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
		this.output(tembd, context);
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
	
	public void back(Tensor delta,Tensor tDiff) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(tDiff);

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		resnetFirst.get(0).update();
		
		if(tEmbDim > 0) {
			tEmbLayers.get(0).update();
		}
		
		
		resnetSecond.get(0).update();
		
		residualInputs.get(0).update();
		
		for(int i = 0;i<numLayers;i++) {

			attns.get(i).update();
			
			if(crossAttn) {
				contextProjs.get(i).update();
				crossAttns.get(i).update();
			}
			
			resnetFirst.get(i+1).update();
			
			if(tEmbDim > 0) {
				tEmbLayers.get(i+1).update();
			}
			
			
			resnetSecond.get(i+1).update();
			
			residualInputs.get(i+1).update();
			
		}

	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.unet_mid;
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
		
		resnetFirst.get(0).accGrad(scale);
		
		if(tEmbDim > 0) {
			tEmbLayers.get(0).accGrad(scale);
		}
		
		
		resnetSecond.get(0).accGrad(scale);
		
		residualInputs.get(0).accGrad(scale);
		
		for(int i = 0;i<numLayers;i++) {
			
			resnetFirst.get(i+1).accGrad(scale);
			
			if(tEmbDim > 0) {
				tEmbLayers.get(i+1).accGrad(scale);
			}
			
			resnetSecond.get(i+1).accGrad(scale);
			
			residualInputs.get(i+1).accGrad(scale);
			
			attns.get(i).accGrad(scale);
			
			if(crossAttn) {
				contextProjs.get(i).accGrad(scale);
				crossAttns.get(i).accGrad(scale);
			}
			
		}

	}

}
