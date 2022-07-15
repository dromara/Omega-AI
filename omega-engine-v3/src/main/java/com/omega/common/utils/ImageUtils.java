package com.omega.common.utils;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import com.omega.engine.nn.data.ImageData;

@Component
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
	public float[][][][] getImageGrayPixelToVector(String image) throws Exception {
		
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
		float[][][][] gray = new float[1][1][1][width * height];
		System.out.println("width=" + width + ",height=" + height + ".");
		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
		for (int i = miny; i < height; i++) {
			for (int j = minx; j < width; j++) {
				int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				int r = (pixel & 0xff0000) >> 16;
				int g = (pixel & 0xff00) >> 8;
				int b = (pixel & 0xff);
				gray[0][0][0][i*j + j] = (float) (r * 0.3 + g * 0.59 + b * 0.11);
			}
		}
		
		return gray;
	}
	
	/**
	 * 读取一张图片的RGB值
	 * 
	 * @throws Exception
	 */
	public float[][][][] getImageGrayPixel(String image) throws Exception {
		
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
		float[][][][] gray = new float[1][1][height][width];
		System.out.println("width=" + width + ",height=" + height + ".");
		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
		for (int i = miny; i < height; i++) {
			for (int j = minx; j < width; j++) {
				int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				int r = (pixel & 0xff0000) >> 16;
				int g = (pixel & 0xff00) >> 8;
				int b = (pixel & 0xff);
				gray[0][0][i][j] = (float) (r * 0.3 + g * 0.59 + b * 0.11);
			}
		}
		
		return gray;
	}
	
	/**
	 * 读取一张图片的RGB值
	 * 
	 * @throws Exception
	 */
	public float[][][][] getImageGrayPixelToVector(File input,boolean normalization) throws Exception {
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(input);
		} catch (Exception e) {
			e.printStackTrace();
		}
		int width = bi.getWidth();
		int height = bi.getHeight();
		int minx = bi.getMinX();
		int miny = bi.getMinY();
		float[][][][] gray = new float[1][1][1][width * height];
		System.out.println("width=" + width + ",height=" + height + ".");
		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
		for (int i = miny; i < height; i++) {
			for (int j = minx; j < width; j++) {
				int rgb = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >> 8) & 0xFF;
				int b = (rgb >> 0) & 0xFF;
				if(normalization) {
					gray[0][0][0][i*j + j] = (float) ((r * 0.3 + g * 0.59 + b * 0.11) / 255);
				}else {
					gray[0][0][0][i*j + j] = (float) (r * 0.3 + g * 0.59 + b * 0.11);
				}
			}
		}
		
		return gray;
	}
	
	/**
     * 图片灰度化的方法
     * @param status 灰度化方法的种类，1表示最大值法，2表示最小值法，3表示均值法，4加权法
     * @param imagePath 需要灰度化的图片的位置
     * @param outPath 灰度化处理后生成的新的灰度图片的存放的位置
     * @throws IOException
     */
    public void grayImage(File file, String outPath) throws IOException {
        BufferedImage image = ImageIO.read(file);
 
        int width = image.getWidth();
        int height = image.getHeight();
        
        BufferedImage grayImage = new BufferedImage(width, height,  image.getType());
        //BufferedImage grayImage = new BufferedImage(width, height,  BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
            	int argb = image.getRGB(i, j);
				int r = (argb & 0xff0000) >> 16;
				int g = (argb & 0xff00) >> 8;
				int b = (argb & 0xff);
				int gray = (int) (r * 0.3 + g * 0.59 + b * 0.11);
//                System.out.println("像素坐标：" + " x=" + i + "   y=" + j + "   rgb=" + argb + "    r=" + r + "   灰度值=" + gray);

                grayImage.setRGB(i, j, colorToRGB(255, gray, gray, gray));
            }
        }
        File newFile = new File(outPath);
        ImageIO.write(grayImage, "png", newFile);
    }
    
	/**
     * 图片灰度化的方法
     * @param status 灰度化方法的种类，1表示最大值法，2表示最小值法，3表示均值法，4加权法
     * @param imagePath 需要灰度化的图片的位置
     * @param outPath 灰度化处理后生成的新的灰度图片的存放的位置
     * @throws IOException
     */
    public void grayImage(MultipartFile file, String outPath) throws IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
 
        int width = image.getWidth();
        int height = image.getHeight();
        
        BufferedImage grayImage = new BufferedImage(width, height,  image.getType());
        //BufferedImage grayImage = new BufferedImage(width, height,  BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
            	int argb = image.getRGB(i, j);
				int r = (argb & 0xff0000) >> 16;
				int g = (argb & 0xff00) >> 8;
				int b = (argb & 0xff);
				int gray = (int) (r * 0.3 + g * 0.59 + b * 0.11);
                System.out.println("像素坐标：" + " x=" + i + "   y=" + j + "   rgb=" + argb + "    r,g,b=" + r + "," + g + "," + b + "   灰度值=" + gray);

                grayImage.setRGB(i, j, colorToRGB(255, gray, gray, gray));
            }
        }
        File newFile = new File(outPath);
        ImageIO.write(grayImage, "png", newFile);
    }
	
    private int colorToRGB(int alpha, int red, int green, int blue) {
    	 
        int newPixel = 0;
        newPixel += alpha;
        newPixel = newPixel << 8;
        newPixel += red;
        newPixel = newPixel << 8;
        newPixel += green;
        newPixel = newPixel << 8;
        newPixel += blue;
 
        return newPixel;
 
    }
    
	/**
	 * 读取一张图片的RGB值
	 * 
	 * @throws Exception
	 */
	public float[][][][] getImageGrayPixel(InputStream input,boolean normalization) throws Exception {
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(input);
		} catch (Exception e) {
			e.printStackTrace();
		}
		int width = bi.getWidth();
		int height = bi.getHeight();
		float[][][][] gray = new float[1][1][height][width];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
				int r = (pixel & 0xff0000) >> 16;
				int g = (pixel & 0xff00) >> 8;
				int b = (pixel & 0xff);
				
				int grayInt = (int) (r * 0.3 + g * 0.59 + b * 0.11);
				
				if(normalization) {
					gray[0][0][j][i] = (float) ((grayInt * 1.0d) / 255.0d);
				}else {
					gray[0][0][j][i] = grayInt;
				}

			}
		}
		
		return gray;
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
	 * red 1 green 2 bule 3
	 * @param path
	 * @param extName
	 * @param rgb
	 * @param rgbType
	 * @return
	 */
	public boolean createRGBImage(String path,String extName,float[][] rgb,int rgbType) {
		
		int[][] rgbInt = new int[rgb.length][rgb[0].length];
		
		for(int i = 0;i<rgb.length;i++) {
			for(int j = 0;j<rgb[i].length;j++) {
				if(rgbType == 1) { //red
					rgbInt[i][j] = (int) rgb[i][j] << 16;
				}else if(rgbType == 2) {
					rgbInt[i][j] = (int) rgb[i][j] << 8;
				}else {
					rgbInt[i][j] = (int) rgb[i][j];
				}
				
			}
		}
		
		BufferedImage bufferedImage = this.convertRGBImage(rgbInt);
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
