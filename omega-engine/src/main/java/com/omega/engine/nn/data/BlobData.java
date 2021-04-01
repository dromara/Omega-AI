package com.omega.engine.nn.data;

/**
 * BlobData
 * @author Administrator
 *
 */
public class BlobData extends BaseData {
	
	public int number = 0;
	
	public int channel = 0;
	
	public int width = 0;
	
	public int height = 0;
	
	public double[][][][] dataInput;
	
	public double[][] dataLabel;
	
	public String[] labels;
	
	public String[] labelSet;
	
	public BlobData(int number,int channel,int width,int height,double[][][][] dataInput,double[][] dataLabel,String[] labels,String[] labelSet) {
		this.number = number;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.dataInput = dataInput;
		this.dataLabel = dataLabel;
		this.labels = labels;
		this.labelSet = labels;
	}

}
