package com.omega.engine.nn.data;

import com.omega.common.utils.MatrixUtils;

/**
 * Blob
 * @author Administrator
 *
 */
public class Blob {
	
	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;

	public int width = 0;
	
	public int dataLength = 0;
	
	/**
	 * number * channel * height * width
	 */
	public double[][][][] maxtir;
	
	public double[][] labels;
	
	public String[] label;
	
	public Blob(int number,int channel,int height,int width) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.maxtir = MatrixUtils.zero(this.number, this.channel, this.height, this.width);
	}
	
	public Blob(int number,int channel,int height,int width,double[][] data) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.maxtir = MatrixUtils.transform(data, this.number,this.channel, this.height, this.width);
	}
	
	public Blob(int number,int channel,int height,int width,double[][][][] maxtir) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.maxtir = maxtir;
		this.dataLength = number * channel * height * width;
	}
	
	public Blob(double[][][][] maxtir) {
		this.number = maxtir.length;
		this.channel = maxtir[0].length;
		this.height = maxtir[0][0].length;
		this.width = maxtir[0][0][0].length;
		this.maxtir = maxtir;
		this.dataLength = number * channel * height * width;
	}
	
	public void clear() {
		this.maxtir = MatrixUtils.zero(this.number, this.channel, this.height, this.width);
	}
	
}
