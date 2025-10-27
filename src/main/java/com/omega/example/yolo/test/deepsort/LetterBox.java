package com.omega.example.yolo.test.deepsort;

import java.awt.image.BufferedImage;

/**
 * 
 */
public class LetterBox {

	private int width;
	private int height;
	private int dw;
	private int dh;
	private double ratio;
	
	private BufferedImage image;
	
	public int getWidth() {
		return width;
	}
	public void setWidth(int width) {
		this.width = width;
	}
	public int getHeight() {
		return height;
	}
	public void setHeight(int height) {
		this.height = height;
	}
	public int getDw() {
		return dw;
	}
	public void setDw(int dw) {
		this.dw = dw;
	}
	public int getDh() {
		return dh;
	}
	public void setDh(int dh) {
		this.dh = dh;
	}
	public double getRatio() {
		return ratio;
	}
	public void setRatio(double ratio) {
		this.ratio = ratio;
	}
	public BufferedImage getImage() {
		return image;
	}
	public void setImage(BufferedImage image) {
		this.image = image;
	}
	
	
}
