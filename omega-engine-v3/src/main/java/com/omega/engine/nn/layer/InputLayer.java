package com.omega.engine.nn.layer;

import com.omega.engine.nn.data.Blob;

public class InputLayer extends Layer {
	
	public InputLayer(int channel,int height,int width) {
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = width;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.input = this.network.inputData;
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		this.output = this.input;
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
		this.init();
		this.output();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("input layer not have diff.");
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.input;
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
	
}
