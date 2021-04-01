package com.omega.engine.nn.data;

/**
 * data set
 * @author Administrator
 *
 */
public class DataSet extends BaseData{
	
	public int inputSize = 0;
	
	public int labelSize = 0;

	public DataSet(int dataSize,int inputSize,int labelSize,double[][] dataInput,double[][] dataLabel,String[] labels,String[] labelSet) {
		this.dataInput = dataInput;
		this.dataLabel = dataLabel;
		this.dataSize = dataSize;
		this.inputSize = inputSize;
		this.labelSize = labelSize;
		this.labels = labels;
		this.labelSet = labelSet;
	}

}
