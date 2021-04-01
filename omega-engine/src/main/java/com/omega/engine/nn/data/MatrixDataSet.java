package com.omega.engine.nn.data;

/**
 * MatrixDataSet
 * @author Administrator
 *
 */
public class MatrixDataSet extends BaseData {
	
	public int channel = 0;
	
	public int width = 0;
	
	public int height = 0;
	
	public int labelSize = 0;
	
	public double[][][][] matrixDataInput;
	
	public MatrixDataSet(int dataSize,int channel,int height,int width,int labelSize,double[][][][] matrixDataInput,double[][] dataInput,double[][] dataLabel,String[] labels,String[] labelSet) {
		this.matrixDataInput = matrixDataInput;
		this.dataInput = dataInput;
		this.dataLabel = dataLabel;
		this.dataSize = dataSize;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.labelSize = labelSize;
		this.labels = labels;
		this.labelSet = labelSet;
	}
	
}
