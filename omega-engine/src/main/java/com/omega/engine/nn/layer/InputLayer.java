package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;

public class InputLayer extends Layer {
	
	public int channel = 0;
	
	public int height = 0;
	
	public int width = 0;
	
	public InputLayer(int inputNum) {
		this.index = 0;
		this.inputNum = inputNum;
		this.outputNum = inputNum;
		this.layerType = LayerType.input;
	}
	
	public InputLayer(int inputNum,int channelSize,int h,int w) {
		this.index = 0;
		this.inputNum = inputNum;
		this.outputNum = inputNum;
		this.layerType = LayerType.input;
		this.channel = channelSize;
		this.height = h;
		this.width = w;
		this.inputShape = new int[] {channelSize, h, w};
		this.outputShape = new int[] {channelSize, h, w};
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	public void input(double[] data) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(data);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		this.output = this.input;
	}
	
	@Override
	public void active() {
		// TODO Auto-generated method stub
		this.active = this.input;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.output();
		this.active();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.input;
	}

	@Override
	public double[] activeTemp(double[] output) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] diffTemp() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

}
