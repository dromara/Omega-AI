package com.omega.engine.nn.data;

/**
 * data set
 * @author Administrator
 *
 */
public class DataSet extends BaseData{
	
	public int inputSize = 0;
	
	public int labelSize = 0;

	public DataSet(int number,int channel,int height,int width,int labelSize,float[][] dataInput,float[][] dataLabel,String[] labels,String[] labelSet) {
		this.dataInput = dataInput;
		this.dataLabel = dataLabel;
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.inputSize = number * channel * height * width;
		this.labelSize = labelSize;
		this.labels = labels;
		this.labelSet = labelSet;
		this.input = Blobs.blob(number, channel, height, width, dataInput);
	}
	
	public DataSet(int number,int channel,int height,int width,int labelSize,float[][][][] dataInput,float[][] dataLabel,String[] labels,String[] labelSet) {
		this.dataLabel = dataLabel;
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.inputSize = number * channel * height * width;
		this.labelSize = labelSize;
		this.labels = labels;
		this.labelSet = labelSet;
		this.input = Blobs.blob(number, channel, height, width, dataInput);
	}
	
}
