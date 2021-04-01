package com.omega.common.utils;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import com.omega.engine.nn.data.ImageData;

public class ImageUtils {
	
	/**
	 * 读取一张图片的RGB值
	 * 
	 * @throws Exception
	 */
	public void getImagePixel(String image) throws Exception {
		int[] rgb = new int[3];
		File file = new File(image);
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		int width = bi.getWidth();
		int height = bi.getHeight();
		int minx = bi.getMinX();
		int miny = bi.getMinY();
		System.out.println("width=" + width + ",height=" + height + ".");
		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
		for (int i = minx; i < width; i++) {
			for (int j = miny; j < height; j++) {
				int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				rgb[0] = (pixel & 0xff0000) >> 16;
				rgb[1] = (pixel & 0xff00) >> 8;
				rgb[2] = (pixel & 0xff);
				System.out.println("i=" + i + ",j=" + j + ":(" + rgb[0] + ","
						+ rgb[1] + "," + rgb[2] + ")");
			}
		}
	}
	
	/**
	 * 读取一张图片的RGB值
	 * 
	 * @throws Exception
	 */
	public ImageData getImageData(String image) throws Exception {
		
		File file = new File(image);
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		int width = bi.getWidth();
		int height = bi.getHeight();
		int minx = bi.getMinX();
		int miny = bi.getMinY();
//		System.out.println("width=" + width + ",height=" + height + ".");
//		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
//		
		int[][] r = new int[width][height];
		int[][] g = new int[width][height];
		int[][] b = new int[width][height];
		
		for (int i = minx; i < width; i++) {
			for (int j = miny; j < height; j++) {
				int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				r[i][j] = (pixel & 0xff0000) >> 16;
				g[i][j] = (pixel & 0xff00) >> 8;
				b[i][j] = (pixel & 0xff);
			}
		}
		
		String extName = file.getName().substring(file.getName().lastIndexOf("."));

		extName = extName.replace(".", "");
		
		ImageData data = new ImageData(width, height, r, g, b,file.getName(),extName);

		return data;
	}
	
	public BufferedImage convertRGBImage(int[][] rgbValue){
		int height = rgbValue.length;
		int width = rgbValue[0].length;
		BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for(int y=0;y<height;y++){
			for(int x=0;x<width;x++){
				bufferedImage.setRGB(x, y, rgbValue[y][x]);
			}
		}
		return bufferedImage;
	}
	
	public boolean createRGBImage(String path,String extName,int[][] rgb) {
		
		BufferedImage bufferedImage = this.convertRGBImage(rgb);
		System.out.println(extName);
		File outputfile = new File(path);
		try {
			System.out.println(ImageIO.write(bufferedImage, extName, outputfile));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	/**
	 * 返回屏幕色彩值
	 * 
	 * @param x
	 * @param y
	 * @return
	 * @throws AWTException
	 */
	public int getScreenPixel(int x, int y) throws AWTException { // 函数返回值为颜色的RGB值。
		Robot rb = null; // java.awt.image包中的类，可以用来抓取屏幕，即截屏。
		rb = new Robot();
		Toolkit tk = Toolkit.getDefaultToolkit(); // 获取缺省工具包
		Dimension di = tk.getScreenSize(); // 屏幕尺寸规格
		System.out.println(di.width);
		System.out.println(di.height);
		Rectangle rec = new Rectangle(0, 0, di.width, di.height);
		BufferedImage bi = rb.createScreenCapture(rec);
		int pixelColor = bi.getRGB(x, y);
 
		return 16777216 + pixelColor; // pixelColor的值为负，经过实践得出：加上颜色最大值就是实际颜色值。
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		
		String testPath = "E:\\2.png";
		
		String testOutPath = "E:\\2_R.png";
		
		ImageUtils rc = new ImageUtils();
		
		ImageData data =  rc.getImageData(testPath);
		
		rc.createRGBImage(testOutPath, data.getExtName(), data.getR());
		
	}

	
}
