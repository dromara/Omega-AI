package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.pooling.PoolingType;

/**
 * PoolingLayer
 * 
 * @author Administrator
 *
 */
public class PoolingLayer extends Layer {
	
	public PoolingType poolingType;
	
	public int pWidth = 0;
	
	public int pHeight = 0;
	
	public int stride = 1;
	
	public double[][][][] pInput;  //n * c * h * w
	
	public double[][][][][] mask;
	
	public PoolingLayer(int channel,int width,int height,int pWidth,int pHeight,int stride,PoolingType poolingType) {
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.pWidth = pWidth;
		this.pHeight = pHeight;
		this.stride = stride;
		this.poolingType = poolingType;
		initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		this.mask = MatrixUtils.zero(this.number,this.channel, this.oHeight * this.oWidth, this.pHeight, this.pWidth);
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.channel;
		this.oWidth = (this.width - pWidth) / this.stride + 1;
		this.oHeight = (this.height - pHeight) / this.stride + 1;
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		this.output.maxtir = MatrixOperation.poolingAndMask(this.input.maxtir, this.mask,this.pWidth, this.pHeight, this.stride, this.poolingType);
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		this.diff.maxtir =  MatrixOperation.poolingDiff(this.delta.maxtir, this.mask, this.diff.maxtir, this.pWidth, this.pHeight, this.stride);
		
//		System.out.println("pooling layer ["+this.index+"]");
//		
//		MatrixOperation.printImage(this.nextDiff);
//		
//		System.out.println("pooling layer ["+this.index+"] end.");
		
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
		
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		if(this.network.GRADIENT_CHECK) {
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
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("pooling layer["+this.index+"]diff start:");
//		
//		MatrixOperation.printImage(this.diff);
//		
//		System.out.println("pooling layer["+this.index+"]diff end.");
	}

	@Override
	public double[][][][] output(double[][][][] input) {
		// TODO Auto-generated method stub
		double[][][][] output = MatrixOperation.poolingAndMask(input, this.mask,this.pWidth, this.pHeight, this.stride, this.poolingType);
		return output;
	}
	
}
