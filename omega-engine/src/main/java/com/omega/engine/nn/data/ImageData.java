package com.omega.engine.nn.data;

public class ImageData {
	
	private int weight;
	
	private int height;
	
	private int[][] r;
	
	private int[][] g;
	
	private int[][] b;
	
	private String label;
	
	private String fileName;
	
	private String extName;
	
	public ImageData() {
		
	}
	
	public ImageData(int weight,int height,int[][] r,int[][] g,int[][] b) {
		this.weight = weight;
		this.height = height;
		this.r = r;
		this.g = g;
		this.b = b;
	}
	
	public ImageData(int weight,int height,int[][] r,int[][] g,int[][] b,String fileName,String extName) {
		this.weight = weight;
		this.height = height;
		this.r = r;
		this.g = g;
		this.b = b;
		this.fileName = fileName;
		this.extName = extName;
	}
	
	public ImageData(int weight,int height,int[][] r,int[][] g,int[][] b,String label) {
		this.weight = weight;
		this.height = height;
		this.r = r;
		this.g = g;
		this.b = b;
		this.label = label;
	}
	
	public ImageData(int weight,int height,int[][] r,int[][] g,int[][] b,String fileName,String extName,String label) {
		this.weight = weight;
		this.height = height;
		this.r = r;
		this.g = g;
		this.b = b;
		this.label = label;
	}
	
	public int getWeight() {
		return weight;
	}

	public void setWeight(int weight) {
		this.weight = weight;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int[][] getR() {
		return r;
	}

	public void setR(int[][] r) {
		this.r = r;
	}

	public int[][] getG() {
		return g;
	}

	public void setG(int[][] g) {
		this.g = g;
	}

	public int[][] getB() {
		return b;
	}

	public void setB(int[][] b) {
		this.b = b;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fileName) {
		this.fileName = fileName;
	}

	public String getExtName() {
		return extName;
	}

	public void setExtName(String extName) {
		this.extName = extName;
	}

}
