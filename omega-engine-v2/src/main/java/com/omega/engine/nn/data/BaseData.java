package com.omega.engine.nn.data;

public abstract class BaseData {
	
	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;
	
	public int width = 0;

	public double[][] dataInput;
	
	public Blob input;
	
	public double[][] dataLabel;
	
	public String[] labels;
	
	public String[] labelSet;
	
	public Blob getRandomData(int[] indexs) {
		Blob data = Blobs.blob(indexs.length, channel, height, width);
		data.labels = new double[indexs.length][dataLabel[0].length];
		for(int i = 0;i<indexs.length;i++) {
			int index = indexs[i];
			data.maxtir[i] = this.input.maxtir[index];
			data.labels[i] = this.dataLabel[index];
		}
		return data;
	}
	
	public Blob getOnceData(int index) {
		Blob data = Blobs.blob(1, channel, height, width);
		data.labels = new double[1][dataLabel[index].length];
		data.maxtir[0] = this.input.maxtir[index];
		data.labels[0] = this.dataLabel[index];
		return data;
	}
	
}
