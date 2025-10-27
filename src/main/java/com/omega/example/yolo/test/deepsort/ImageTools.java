package com.omega.example.yolo.test.deepsort;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

/**
 * 绘图工具
 */
public class ImageTools {

	private static Font defaultFont = new Font("Monospaced", Font.PLAIN, 14);
	private static BasicStroke defaultBasicStroke = new BasicStroke(1f);
	
	static {

	}

	/**
	 * 设置字体大小
	 * 
	 * @param fontSize
	 */
	public static void setFontSize(int fontSize) {
		defaultFont = defaultFont.deriveFont(Font.PLAIN, fontSize);
	}
	
	/**
	 * 设置线宽
	 * @param width
	 */
	public static void setStrokeWidth(float width) {
		defaultBasicStroke = new BasicStroke(width);
	}

	/**
	 * 绘制文字
	 * 
	 * @param image
	 * @param text
	 * @param x
	 * @param y
	 */
	public static void drawText(BufferedImage image, String text, int x, int y, Color color) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(defaultFont);
		g.setStroke(defaultBasicStroke);
		g.drawString(text, x, y);
		g.dispose();

	}
	
	/**
	 * 绘制文字
	 * 
	 * @param image
	 * @param text
	 * @param x
	 * @param y
	 */
	public static void drawText(BufferedImage image, String text, int x, int y, Color color, Font font) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(null == font ? defaultFont : font);
		g.drawString(text, x, y);
		g.dispose();

	}

	/**
	 * 绘制矩形
	 * 
	 * @param image
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 * @param color
	 */
	public static void drawRect(BufferedImage image, int x, int y, int width, int height, Color color) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(defaultFont);
		g.setStroke(defaultBasicStroke);
		g.drawRect(x, y, width, height);
		g.dispose();

	}
	
	public static void drawRect(BufferedImage image, int x, int y, int width, int height, Color color, float strokeWidth) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(defaultFont);
		g.setStroke(new BasicStroke(strokeWidth));
		g.drawRect(x, y, width, height);
		g.dispose();

	}

	/**
	 * 绘制圆点
	 * 
	 * @param image
	 * @param text
	 * @param x
	 * @param y
	 * @param color
	 */
	public static void drawCircle(BufferedImage image, int x, int y, int radius, Color color) {

		drawCircle(image, x, y, radius, color, false);

	}

	/**
	 * 绘制圆点
	 * 
	 * @param image
	 * @param x
	 * @param y
	 * @param radius
	 * @param color
	 * @param fill   是否填充
	 */
	public static void drawCircle(BufferedImage image, int x, int y, int radius, Color color, boolean fill) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(defaultFont);
		if (fill) {
			g.fillOval(x - radius, y - radius, 2 * radius, 2 * radius);
		} else {
			g.drawOval(x - radius, y - radius, 2 * radius, 2 * radius);
		}
		g.dispose();

	}

	/**
	 * 绘制标记
	 * 
	 * @param image
	 * @param text
	 * @param x
	 * @param y
	 * @param color
	 */
	public static void drawMarks(BufferedImage image, String text, int x, int y, Color color) {

		Graphics2D g = (Graphics2D) image.getGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//		AlphaComposite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f);
//		g.setComposite(composite);

		g.setColor(null == color ? Color.RED : color);
//		g.setFont(new Font("Symbola", Font.PLAIN, 20));
//		g.setFont(new Font("Arial", Font.PLAIN, 20));
		g.setFont(defaultFont);
		g.drawString(text, x, y);
		g.dispose();

	}

	/**
	 * 展示图片
	 * 
	 * @param image
	 */
	public static void showImage(BufferedImage image) {
		JFrame frame = new JFrame("图片展示");
		JLabel jLabel = new JLabel(new ImageIcon(image));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(new BorderLayout());
		frame.getContentPane().add(jLabel);
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		frame.setMinimumSize(new Dimension(Math.round(screenSize.width / 2.4f), Math.round(screenSize.height / 1.8f)));
		frame.setMaximumSize(screenSize);
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}

	public static LetterBox letterboxObj(BufferedImage bufferedImage, int width, int height) {
		return letterbox(bufferedImage, width, height, new Color(114, 114, 114));
	}

	/**
	 * 使得图像居中
	 * 
	 * @param bufferedImage
	 * @param width
	 * @param height
	 * @return
	 * @throws Exception
	 */
	public static BufferedImage letterbox(BufferedImage bufferedImage, int width, int height) {
		return letterbox(bufferedImage, width, height, Color.BLACK).getImage();
	}

	/**
	 * 使得图像居中
	 * 
	 * @param bufferedImage
	 * @param width
	 * @param height
	 * @param backColor
	 * @return
	 * @throws Exception
	 */
	public static LetterBox letterbox(BufferedImage bufferedImage, int width, int height, Color backColor) {
		int imgWidth = width;
		int imgHeight = height;

		BufferedImage tempImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics2D = tempImg.createGraphics();
		graphics2D.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
		graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
		graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		if (backColor != null) {
			graphics2D.setBackground(backColor);
			graphics2D.clearRect(0, 0, width, height);
		}
		double constrainRatio = (double) width / (double) height;
		double imageRatio = (double) bufferedImage.getWidth() / (double) bufferedImage.getHeight();
		if (constrainRatio < imageRatio) {
			imgHeight = (int) (width / imageRatio);
		} else {
			imgWidth = (int) (height * imageRatio);
		}

		int dx = (width - imgWidth) / 2;
		int dy = (height - imgHeight) / 2;

//		try {
//			bufferedImage = ImageUtils.scale(bufferedImage, imgWidth, imgHeight);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
		graphics2D.drawImage(bufferedImage, dx, dy, imgWidth, imgHeight, null);
		graphics2D.dispose();
		bufferedImage = tempImg;

		LetterBox letterBox = new LetterBox();
		letterBox.setDh(dy);
		letterBox.setDw(dx);
		letterBox.setHeight(height);
		letterBox.setWidth(width);
		letterBox.setImage(tempImg);
		letterBox.setRatio(constrainRatio);

		return letterBox;
	}

	/**
	 * 获取Pixels
	 * @param image
	 * @return
	 */
	public static float[] getPixels(BufferedImage image) {
		int cols = image.getWidth();
		int rows = image.getHeight();
		Raster data = image.getData();
		int channels = data.getNumBands();
		float[] pixels = new float[channels * rows * cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double[] pixel = new double[channels];
				data.getPixel(i, j, pixel);
				for (int k = 0; k < channels; k++) {
					// 这样设置相当于同时做了image.transpose((2, 0, 1))操作
					pixels[rows * cols * k + j * cols + i] = (float) pixel[k] / 255.0f;
				}
			}
		}

		return pixels;

	}
}
