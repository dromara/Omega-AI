package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.Updater;

/**
 * PoolingLayer
 * 
 * @author Administrator
 *
 */
public class PoolingLayer extends Layer {
	
	public PoolingType poolingType;
	
	public int channel = 0;

	public int width = 0;
	
	public int height = 0;
	
	public int pWidth = 0;
	
	public int pHeight = 0;
	
	public int stride = 1;
	
	public int oChannel = 0;
	
	public int oWidth = 0;
	
	public int oHeight = 0;
	
	public double[][][] input;  //c * h * w
	
	public double[][][] pInput;  //c * h * w
	
	public double[][][] output;  //oc * oh * ow
	
	public double[][][] active;  //oc * oh * ow
	
	public double[][][] diff;
	
	public double[][][] delta;
	
	public double[][][] nextDiff;
	
	public double[][][][] mask;
	
	public PoolingLayer(int channel,int width,int height,int pWidth,int pHeight,int stride,PoolingType poolingType,Updater updater) {
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.pWidth = pWidth;
		this.pHeight = pHeight;
		this.stride = stride;
		this.poolingType = poolingType;
		this.updater = updater;
		initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.zero(this.oChannel, this.oHeight, this.oWidth);
		this.mask = MatrixOperation.zero(this.channel, this.oHeight * this.oWidth, this.pHeight, this.pWidth);
		this.diff = MatrixOperation.zero(this.channel, this.height, this.width);
		this.nextDiff = MatrixOperation.zero(this.oChannel, this.oHeight, this.oWidth);
		this.delta = MatrixOperation.zero(this.oChannel, this.oHeight, this.oWidth);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.channel;
		this.oWidth = (this.width - pWidth) / this.stride + 1;
		this.oHeight = (this.height - pHeight) / this.stride + 1;
	}

	public void input(double[][][] data) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(data);
	}

	public void nextDiff(double[][][] data) {
		// TODO Auto-generated method stub
		this.nextDiff = MatrixOperation.clone(data);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.poolingAndMask(this.input, this.mask,this.pWidth, this.pHeight, this.stride, this.poolingType);
	}

	@Override
	public void active() {
		// TODO Auto-generated method stub
		this.active = this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * 计算当前层梯度
		 */
		this.delta = this.nextDiff;
		
		this.diff =  MatrixOperation.poolingDiff(this.delta, this.mask, this.diff, this.pWidth, this.pHeight, this.stride);
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		this.output();
		this.active();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.diff();
		if(this.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.pooling;
	}

	@Override
	public double[] activeTemp(double[] output) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public double[] diffTemp() {
		// TODO Auto-generated method stub
		return MatrixOperation.transform(this.diff);
	}

	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return MatrixOperation.transform(this.output);
	}
	
}
