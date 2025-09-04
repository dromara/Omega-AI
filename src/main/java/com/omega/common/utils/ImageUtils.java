package com.omega.common.utils;

import com.madgag.gif.fmsware.AnimatedGifEncoder;
import com.omega.engine.nn.data.ImageData;
import com.omega.example.yolo.utils.OMImage;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Comparator;

public class ImageUtils {
    public static float[] mean = new float[]{0.491f, 0.482f, 0.446f};
    public static float[] std = new float[]{0.247f, 0.243f, 0.261f};
    public static Color[] colors = new Color[]{Color.red, Color.blue, Color.green, Color.yellow, Color.white, Color.gray, Color.pink, Color.orange};

    private static int colorToRGB(int alpha, int red, int green, int blue) {
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
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
        return outputImage;
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage resizeImageFromBicubic(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) outputImage.getGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        return outputImage;
    }

    public static BufferedImage getScaledInstance(BufferedImage img, int targetWidth, int targetHeight, Object hint, boolean higherQuality) {
        int type = (img.getTransparency() == Transparency.OPAQUE) ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
        BufferedImage ret = (BufferedImage) img;
        int w, h;
        if (higherQuality) {
            // Use multi-step technique: start with original size, then
            // scale down in multiple passes with drawImage()
            // until the target size is reached
            w = img.getWidth();
            h = img.getHeight();
        } else {
            // Use one-step technique: scale directly from original
            // size to target size with a single drawImage() call
            w = targetWidth;
            h = targetHeight;
        }
        do {
            if (higherQuality && w > targetWidth) {
                w /= 2;
                if (w < targetWidth) {
                    w = targetWidth;
                }
            }
            if (higherQuality && h > targetHeight) {
                h /= 2;
                if (h < targetHeight) {
                    h = targetHeight;
                }
            }
            BufferedImage tmp = new BufferedImage(w, h, type);
            Graphics2D g2 = tmp.createGraphics();
            g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, hint);
            g2.drawImage(ret, 0, 0, w, h, null);
            g2.dispose();
            ret = tmp;
        } while (w != targetWidth || h != targetHeight);
        return ret;
    }

    public static BufferedImage resizeImageBicubic2(BufferedImage originalImage, int newWidth, int newHeight) throws IOException {
        BufferedImage outputImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        int oldWidth = originalImage.getWidth();
        int oldHeight = originalImage.getHeight();
        int ox, oy;        // position in old image
        double dx, dy;        // delta_x, delta_y
        double tx = (double) oldWidth / newWidth;
        double ty = (double) oldHeight / newHeight;
        double Bmdx, Bdyn;
        int newPixelR = 0;
        int newPixelG = 0;
        int newPixelB = 0;
        int oxm;
        int oyn;
        int oldPixel;
        double oldPixelR;
        double oldPixelG;
        double oldPixelB;
        for (int ny = 0; ny < newHeight; ny++) {
            for (int nx = 0; nx < newWidth; nx++) {
                newPixelR = 0;
                newPixelG = 0;
                newPixelB = 0;
                ox = (int) (tx * nx);
                oy = (int) (ty * ny);
                dx = tx * nx - ox;
                dy = ty * ny - oy;
                // Bicubic algorithm
                for (int m = -1; m <= 2; m++) {
                    Bmdx = BSpline(m - dx);
                    for (int n = -1; n <= 2; n++) {
                        oxm = ox + m;
                        oyn = oy + n;
                        if (oxm >= 0 && oyn >= 0 && oxm < oldWidth && oyn < oldHeight) {
                            oldPixel = originalImage.getRGB(oxm, oyn);
                            oldPixelR = (oldPixel >> 16) & 0xFF;
                            oldPixelG = (oldPixel >> 8) & 0xFF;
                            oldPixelB = oldPixel & 0xFF;
                            Bdyn = BSpline(dy - n);
                            newPixelR += (int) (oldPixelR * Bmdx * Bdyn);
                            newPixelG += (int) (oldPixelG * Bmdx * Bdyn);
                            newPixelB += (int) (oldPixelB * Bmdx * Bdyn);
                        }
                    }
                }
                System.out.println(nx + ":" + ny + ":" + newPixelR);
                int interpolatedRgb = (newPixelR << 16) | (newPixelG << 8) | newPixelB;
                //                System.err.println(interpolatedRgb);
                outputImage.setRGB(nx, ny, interpolatedRgb);
            }
        }
        return outputImage;
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage resizeImageBicubic(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        //        Interpolation interpolation = new Interpolation();
        int oldWidth = originalImage.getWidth();
        int oldHeight = originalImage.getHeight();
        int ox, oy;        // position in old image
        double dx, dy;        // delta_x, delta_y
        double oldPixelR;
        double oldPixelG;
        double oldPixelB;
        double Bmdx, Bdyn;
        int oxm;
        int oyn;
        int newPixelR = 0;
        int newPixelG = 0;
        int newPixelB = 0;
        int oldPixel;
        for (int ny = 0; ny < targetHeight; ny++) {
            for (int nx = 0; nx < targetWidth; nx++) {
                newPixelR = 0;
                newPixelG = 0;
                newPixelB = 0;
                ox = (int) (targetWidth * nx);
                oy = (int) (targetHeight * ny);
                dx = targetWidth * nx - ox;
                dy = targetHeight * ny - oy;
                // Bicubic algorithm
                for (int m = -1; m <= 2; m++) {
                    Bmdx = BSpline(m - dx);
                    for (int n = -1; n <= 2; n++) {
                        oxm = ox + m;
                        oyn = oy + n;
                        if (oxm >= 0 && oyn >= 0 && oxm < oldWidth && oyn < oldHeight) {
                            oldPixel = originalImage.getRGB(oxm, oyn);
                            oldPixelR = (oldPixel >> 16) & 0xFF;
                            oldPixelG = (oldPixel >> 8) & 0xFF;
                            oldPixelB = oldPixel & 0xFF;
                            System.out.println(oldPixelR);
                            Bdyn = BSpline(dy - n);
                            newPixelR += (int) (oldPixelR * Bmdx * Bdyn);
                            newPixelG += (int) (oldPixelG * Bmdx * Bdyn);
                            newPixelB += (int) (oldPixelB * Bmdx * Bdyn);
                        }
                    }
                }
                int interpolatedRgb = (newPixelR << 16) | (newPixelG << 8) | newPixelB;
//                System.err.println(interpolatedRgb);
                outputImage.setRGB(nx, ny, interpolatedRgb);
            }
        }
        //        for (int x = 0; x < targetWidth; x++) {
        //            for (int y = 0; y < targetHeight; y++) {
        //
        //                double u = x * 1.0d / targetWidth * (width - 1);
        //                double v = y * 1.0d / targetHeight * (height - 1);
        //
        //                int upperLeftX = (int) Math.ceil(u) - 1;
        //                int upperLeftY = (int) Math.ceil(v) - 1;
        //
        //                double[][] redMatrixForPixelXY = new double[4][4];
        //                double[][] greenMatrixForPixelXY = new double[4][4];
        //                double[][] blueMatrixForPixelXY = new double[4][4];
        //
        //                for (int m = 0; m < 4; m++) {
        //                    for (int n = 0; n < 4; n++) {
        //                        int suitableX = checkBounds(upperLeftX + m, 0, width - 1);
        //                        int suitableY = checkBounds(upperLeftY + n, 0, height - 1);
        //                        int rgb = originalImage.getRGB(suitableX, suitableY);
        //
        //                        redMatrixForPixelXY[m][n] = (rgb >> 16) & 0xFF;
        //                        greenMatrixForPixelXY[m][n] = (rgb >> 8) & 0xFF;
        //                        blueMatrixForPixelXY[m][n] = rgb & 0xFF;
        //                    }
        //                }
        //
        //                double xFraction = u - Math.floor(u);
        //                double yFraction = v - Math.floor(v);
        //
        //                int combinedRed = (int) checkBounds(interpolation.twoDimensionalBicubicInterpolation(redMatrixForPixelXY, xFraction, yFraction), 0, 255);
        //                int combinedGreen = (int) checkBounds(interpolation.twoDimensionalBicubicInterpolation(greenMatrixForPixelXY, xFraction, yFraction), 0, 255);
        //                int combinedBlue = (int) checkBounds(interpolation.twoDimensionalBicubicInterpolation(blueMatrixForPixelXY, xFraction, yFraction), 0, 255);
        //
        //                int interpolatedRgb = (combinedRed << 16) | (combinedGreen << 8) | combinedBlue;
        //
        //                outputImage.setRGB(x, y, interpolatedRgb);
        //            }
        //        }
        return outputImage;
    }

    private static double BSpline(double x) {
        if (x < 0.0)
            x = Math.abs(x);
        if (x >= 0.0 && x <= 1.0)
            return (2.0 / 3.0) + 0.5 * Math.pow(x, 3.0) - Math.pow(x, 2.0);
        else if (x > 1.0 && x <= 2.0)
            return 1.0 / 6.0 * Math.pow(2 - x, 3.0);
        return 1.0;
    }

    private static int checkBounds(int value, int min, int max) {
        return Math.round(Math.max(min, Math.min(value, max)));
    }

    private static double checkBounds(double value, double min, double max) {
        return Math.round(Math.max(min, Math.min(value, max)));
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight, int[][] bbox) throws IOException {
        if (originalImage.getWidth() == targetWidth && originalImage.getHeight() == targetHeight) {
            Graphics g = originalImage.getGraphics();
            g.drawImage(originalImage, 0, 0, null);
            if (bbox != null) {
                g.setColor(Color.RED);
                for (int[] box : bbox) {
                    g.setColor(colors[box[0]]);
                    int w = (box[3] - box[1]);
                    int h = (box[4] - box[2]);
                    g.drawRect(box[1], box[2], w, h);
                }
            }
            return originalImage;
        } else {
            Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
            BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
            Graphics g = outputImage.getGraphics();
            g.drawImage(resultingImage, 0, 0, null);
            g.setColor(Color.RED);
            if (bbox != null) {
                for (int[] box : bbox) {
                    g.setColor(colors[box[0]]);
                    int w = (box[3] - box[1]);
                    int h = (box[4] - box[2]);
                    g.drawRect(box[1], box[2], w, h);
                }
            }
            return outputImage;
        }
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight, int[][] bbox, String[] classLabel) throws IOException {
        if (originalImage.getWidth() == targetWidth && originalImage.getHeight() == targetHeight) {
            Graphics g = originalImage.getGraphics();
            g.drawImage(originalImage, 0, 0, null);
            g.setColor(Color.RED);
            if (bbox != null) {
                for (int[] box : bbox) {
                    if (box[0] >= colors.length) {
                        g.setColor(Color.RED);
                    } else {
                        g.setColor(colors[box[0]]);
                    }
                    int w = (box[3] - box[1]);
                    int h = (box[4] - box[2]);
                    g.drawRect(box[1], box[2], w, h);
                    g.drawString(classLabel[box[0]], box[1], box[2]);
                }
            }
            return originalImage;
        } else {
            Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
            BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
            Graphics g = outputImage.getGraphics();
            g.drawImage(resultingImage, 0, 0, null);
            g.setColor(Color.RED);
            if (bbox != null) {
                for (int[] box : bbox) {
                    g.setColor(colors[box[0]]);
                    int w = (box[3] - box[1]);
                    int h = (box[4] - box[2]);
                    g.drawRect(box[1], box[2], w, h);
                    g.drawString(classLabel[box[0]], box[1], box[2]);
                }
            }
            return outputImage;
        }
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage drawImage(BufferedImage originalImage, int targetWidth, int targetHeight, int[][] bbox) throws IOException {
        originalImage.getGraphics().drawImage(originalImage, 0, 0, null);
        originalImage.getGraphics().setColor(Color.BLUE);
        for (int[] box : bbox) {
            int w = (box[3] - box[1]);
            int h = (box[4] - box[2]);
            originalImage.getGraphics().drawRect(box[1], box[2], w, h);
        }
        return originalImage;
    }

    /**
     * 通过BufferedImage图片流调整图片大小
     * <p>
     * 指定压缩后长宽
     */
    public static BufferedImage createRGBImage(BufferedImage originalImage, int[][] bbox) throws IOException {
        originalImage.getGraphics().setColor(Color.red);
        for (int[] box : bbox) {
            int x = box[1];
            int y = box[2];
            originalImage.getGraphics().drawRect(x, y, box[3], box[4]);
        }
        return originalImage;
    }

    public static int[][] color2rgb(float[] data, int height, int width) {
        int[][] rgb = new int[height][width];
        int ocount = height * width;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                int r = (int) data[index];
                int g = (int) data[ocount + index];
                int b = (int) data[ocount * 2 + index];
                //				System.out.println(r);
                int orgb = colorToRGB(255, r, g, b);
                rgb[i][j] = orgb;
            }
        }
        return rgb;
    }

    public static int[][] color2rgb2(float[] data, int height, int width) {
        int[][] rgb = new int[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int r = (int) data[0 * width * height + i * width + j];
                int g = (int) data[1 * width * height + i * width + j];
                int b = (int) data[2 * width * height + i * width + j];
                //				System.out.println(r);
                int orgb = colorToRGB(255, r, g, b);
                rgb[i][j] = orgb;
            }
        }
        return rgb;
    }

    public static int[][] color2rgb2(float[] data, int height, int width, boolean format) {
        int[][] rgb = new int[height][width];
        int ocount = height * width;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                int r = (int) data[index];
                int g = (int) data[ocount + index];
                int b = (int) data[ocount * 2 + index];
                if (format) {
                    r = (int) ((r * std[0] + mean[0]) * 255);
                    g = (int) ((g * std[0] + mean[0]) * 255);
                    b = (int) ((b * std[0] + mean[0]) * 255);
                }
                int orgb = colorToRGB(255, r, g, b);
                rgb[i][j] = orgb;
            }
        }
        return rgb;
    }

    public static int[][] color2rgb2(float[] data, int channel, int height, int width, boolean format) {
        int[][] rgb = new int[height][width];
        int ocount = height * width;
        if (channel > 1) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width + j;
                    int r = (int) data[index];
                    int g = (int) data[ocount + index];
                    int b = (int) data[ocount * 2 + index];
                    if (format) {
                        r = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        g = (int) ((data[ocount + index] * std[0] + mean[0]) * 255 + 0.5f);
                        b = (int) ((data[ocount * 2 + index] * std[0] + mean[0]) * 255 + 0.5f);
                    }
                    r = clamp(r, 0, 255);
                    g = clamp(g, 0, 255);
                    b = clamp(b, 0, 255);
                    int orgb = colorToRGB(255, r, g, b);
                    rgb[i][j] = orgb;
                }
            }
        } else {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width + j;
                    int r = (int) data[index];
                    int g = (int) data[index];
                    int b = (int) data[index];
                    if (format) {
                        r = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        g = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        b = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                    }
                    r = clamp(r, 0, 255);
                    g = clamp(g, 0, 255);
                    b = clamp(b, 0, 255);
                    int orgb = colorToRGB(255, r, g, b);
                    rgb[i][j] = orgb;
                }
            }
        }
        return rgb;
    }

    public static int[][] color2rgb2(float[] data, int channel, int height, int width, boolean format, float[] mean, float[] std) {
        int[][] rgb = new int[height][width];
        int ocount = height * width;
        if (channel > 1) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width + j;
                    int r = (int) data[index];
                    int g = (int) data[ocount + index];
                    int b = (int) data[ocount * 2 + index];
                    if (format) {
                        r = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        g = (int) ((data[ocount + index] * std[0] + mean[0]) * 255 + 0.5f);
                        b = (int) ((data[ocount * 2 + index] * std[0] + mean[0]) * 255 + 0.5f);
                    }
                    r = clamp(r, 0, 255);
                    g = clamp(g, 0, 255);
                    b = clamp(b, 0, 255);
                    int orgb = colorToRGB(255, r, g, b);
                    rgb[i][j] = orgb;
                }
            }
        } else {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width + j;
                    int r = (int) data[index];
                    int g = (int) data[index];
                    int b = (int) data[index];
                    if (format) {
                        r = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        g = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                        b = (int) ((data[index] * std[0] + mean[0]) * 255 + 0.5f);
                    }
                    r = clamp(r, 0, 255);
                    g = clamp(g, 0, 255);
                    b = clamp(b, 0, 255);
                    int orgb = colorToRGB(255, r, g, b);
                    rgb[i][j] = orgb;
                }
            }
        }
        return rgb;
    }

    public static int clamp(int x, int min, int max) {
        if (x > max) {
            x = max;
        } else if (x < min) {
            x = min;
        }
        return x;
    }

    public static int[][] color2rgb(int[] data, int width, int height) {
        int[][] rgb = new int[width][height];
        int ocount = height * width;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int index = i * height + j;
                int r = data[index];
                int g = data[ocount + index];
                int b = data[ocount * 2 + index];
                int orgb = colorToRGB(255, r, g, b);
                rgb[i][j] = orgb;
            }
        }
        return rgb;
    }

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        String testPath = "I:\\000005.jpg";
        ImageUtils rc = new ImageUtils();
        ImageData data = rc.getImageData(testPath);
        String testOutPath = "I:\\_" + data.getFileName();
        rc.createRGBImage(testOutPath, data.getExtName(), color2rgb(data.getColor(), data.getWeight(), data.getHeight()));
    }

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
        //		System.out.println("width=" + width + ",height=" + height + ".");
        //		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
        for (int i = minx; i < width; i++) {
            for (int j = miny; j < height; j++) {
                int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                rgb[0] = (pixel & 0xff0000) >> 16;
                rgb[1] = (pixel & 0xff00) >> 8;
                rgb[2] = (pixel & 0xff);
                System.out.println("i=" + i + ",j=" + j + ":(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")");
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
        //		System.out.println("width=" + width + ",height=" + height + ".");
        //		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
        for (int i = miny; i < height; i++) {
            for (int j = minx; j < width; j++) {
                int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                int r = (pixel & 0xff0000) >> 16;
                int g = (pixel & 0xff00) >> 8;
                int b = (pixel & 0xff);
                gray[0][0][0][i * j + j] = (float) (r * 0.3 + g * 0.59 + b * 0.11);
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
        //		System.out.println("width=" + width + ",height=" + height + ".");
        //		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
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
    public float[][][][] getImageGrayPixelToVector(File input, boolean normalization) throws Exception {
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
        //		System.out.println("width=" + width + ",height=" + height + ".");
        //		System.out.println("minx=" + minx + ",miniy=" + miny + ".");
        for (int i = miny; i < height; i++) {
            for (int j = minx; j < width; j++) {
                int rgb = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = (rgb >> 0) & 0xFF;
                if (normalization) {
                    gray[0][0][0][i * j + j] = (float) ((r * 0.3 + g * 0.59 + b * 0.11) / 255);
                } else {
                    gray[0][0][0][i * j + j] = (float) (r * 0.3 + g * 0.59 + b * 0.11);
                }
            }
        }
        return gray;
    }

    /**
     * 图片灰度化的方法
     *
     * @param status    灰度化方法的种类，1表示最大值法，2表示最小值法，3表示均值法，4加权法
     * @param imagePath 需要灰度化的图片的位置
     * @param outPath   灰度化处理后生成的新的灰度图片的存放的位置
     * @throws IOException
     */
    public void grayImage(File file, String outPath) throws IOException {
        BufferedImage image = ImageIO.read(file);
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage grayImage = new BufferedImage(width, height, image.getType());
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
     * 读取一张图片的RGB值
     *
     * @throws Exception
     */
    public float[][][][] getImageGrayPixel(InputStream input, boolean normalization) throws Exception {
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
                if (normalization) {
                    gray[0][0][j][i] = (float) ((grayInt * 1.0d) / 255.0d);
                } else {
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
        int[][] r = new int[width][height];
        int[][] g = new int[width][height];
        int[][] b = new int[width][height];
        int size = height * width * 3;
        int[] color = new int[size];
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                //				System.out.println(pixel);
                r[i][j] = (pixel & 0xff0000) >> 16;
                g[i][j] = (pixel & 0xff00) >> 8;
                b[i][j] = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r[i][j];
                color[1 * width * height + j * width + i] = g[i][j];
                color[2 * width * height + j * width + i] = b[i][j];
            }
        }
        String extName = file.getName().substring(file.getName().lastIndexOf("."));
        extName = extName.replace(".", "");
        ImageData data = new ImageData(width, height, r, g, b, color, file.getName(), extName);
        return data;
    }

    /**
     * 读取一张图片的RGB值
     *
     * @throws Exception
     */
    public ImageData getImageData(File file) throws Exception {
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
        int[][] r = new int[width][height];
        int[][] g = new int[width][height];
        int[][] b = new int[width][height];
        int size = height * width * 3;
        int[] color = new int[size];
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                //				System.out.println(pixel);
                r[i][j] = (pixel & 0xff0000) >> 16;
                g[i][j] = (pixel & 0xff00) >> 8;
                b[i][j] = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r[i][j];
                color[1 * width * height + j * width + i] = g[i][j];
                color[2 * width * height + j * width + i] = b[i][j];
            }
        }
        String extName = file.getName().substring(file.getName().lastIndexOf("."));
        extName = extName.replace(".", "");
        ImageData data = new ImageData(width, height, r, g, b, color, file.getName(), extName);
        return data;
    }

    /**
     * 读取一张图片的RGB值
     *
     * @throws Exception
     */
    public float[] getImageData(File file, boolean normalization) throws Exception {
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
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 1.0f;
        if (normalization) {
            n = 255.0f;
        }
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r * 1.0f / n;
                color[1 * width * height + j * width + i] = g * 1.0f / n;
                color[2 * width * height + j * width + i] = b * 1.0f / n;
            }
        }
        return color;
    }

    public OMImage loadOMImgAndResize(File file, int tw, int th, float[] mean, float[] std) throws Exception {
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
        if (width != tw || height != th) {
            bi = resizeImage(bi, tw, th);
            width = bi.getWidth();
            height = bi.getHeight();
            minx = bi.getMinX();
            miny = bi.getMinY();
        }
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 255.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                System.out.println(b);
                color[0 * width * height + j * width + i] = (r * 1.0f / n - mean[0]) / std[0];
                color[1 * width * height + j * width + i] = (g * 1.0f / n - mean[1]) / std[1];
                color[2 * width * height + j * width + i] = (b * 1.0f / n - mean[2]) / std[2];
            }
        }
        return new OMImage(3, height, width, color);
    }

    public OMImage loadOMImgAndResizeToBicubic(File file, int tw, int th, float[] mean, float[] std) throws Exception {
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
        if (width != tw || height != th) {
            //			bi = resizeImageBicubic2(bi, tw, th);
            bi = resizeImageFromBicubic(bi, tw, th);
            width = bi.getWidth();
            height = bi.getHeight();
            minx = bi.getMinX();
            miny = bi.getMinY();
        }
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 255.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                //				color[0 * width * height + j * width + i] = r;
                //				color[1 * width * height + j * width + i] = g;
                //				color[2 * width * height + j * width + i] = b;
                color[0 * width * height + j * width + i] = (r * 1.0f / n - mean[0]) / std[0];
                color[1 * width * height + j * width + i] = (g * 1.0f / n - mean[1]) / std[1];
                color[2 * width * height + j * width + i] = (b * 1.0f / n - mean[2]) / std[2];
            }
        }
        return new OMImage(3, height, width, color);
    }

    public OMImage loadOMImage(File file) throws Exception {
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
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 255.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r * 1.0f / n;
                color[1 * width * height + j * width + i] = g * 1.0f / n;
                color[2 * width * height + j * width + i] = b * 1.0f / n;
            }
        }
        return new OMImage(3, height, width, color);
    }

    public OMImage loadOMImage(File file, float[] color) throws Exception {
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
        //		int size = height * width * 3;
        //		float[] color = new float[size];
        float n = 255.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r * 1.0f / n;
                color[1 * width * height + j * width + i] = g * 1.0f / n;
                color[2 * width * height + j * width + i] = b * 1.0f / n;
            }
        }
        return new OMImage(3, height, width, color);
    }

    public OMImage loadOMImage(File file, int maxSize) throws Exception {
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
        //		int size = height * width * 3;
        float[] color = new float[maxSize];
        float n = 255.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                color[0 * width * height + j * width + i] = r * 1.0f / n;
                color[1 * width * height + j * width + i] = g * 1.0f / n;
                color[2 * width * height + j * width + i] = b * 1.0f / n;
            }
        }
        return new OMImage(3, height, width, color);
    }

    public float[] getImageData(File file, boolean normalization, boolean meanStd) throws Exception {
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
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 1.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        if (normalization) {
            n = 255.0f;
        }
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                if (meanStd) {
                    color[0 * width * height + j * width + i] = (float) ((r * 1.0f / n) - mean[0]) / std[0];
                    color[1 * width * height + j * width + i] = (float) ((g * 1.0f / n) - mean[1]) / std[1];
                    color[2 * width * height + j * width + i] = (float) ((b * 1.0f / n) - mean[2]) / std[2];
                } else {
                    color[0 * width * height + j * width + i] = r * 1.0f / n;
                    color[1 * width * height + j * width + i] = g * 1.0f / n;
                    color[2 * width * height + j * width + i] = b * 1.0f / n;
                }
            }
        }
        return color;
    }

    public float[] getImageData(File file, boolean normalization, boolean meanStd, float[] mean, float[] std) throws Exception {
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(file);
        } catch (Exception e) {
            System.err.println("error file:" + file.getName());
            e.printStackTrace();
        }
        int width = bi.getWidth();
        int height = bi.getHeight();
        int minx = bi.getMinX();
        int miny = bi.getMinY();
        int size = height * width * 3;
        float[] color = new float[size];
        float n = 1.0f;
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        if (normalization) {
            n = 255.0f;
        }
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                if (meanStd) {
                    color[0 * width * height + j * width + i] = (float) ((r * 1.0f / n) - mean[0]) / std[0];
                    color[1 * width * height + j * width + i] = (float) ((g * 1.0f / n) - mean[1]) / std[1];
                    color[2 * width * height + j * width + i] = (float) ((b * 1.0f / n) - mean[2]) / std[2];
                } else {
                    color[0 * width * height + j * width + i] = r * 1.0f / n;
                    color[1 * width * height + j * width + i] = g * 1.0f / n;
                    color[2 * width * height + j * width + i] = b * 1.0f / n;
                }
            }
        }
        return color;
    }

    public float[] getImageDataToGray(File file, boolean normalization) throws Exception {
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
        int r = 0;
        int g = 0;
        int b = 0;
        int pixel = 0;
        int size = height * width;
        float[] color = new float[size];
        float n = 1.0f;
        if (normalization) {
            n = 255.0f;
        }
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                int grayInt = (int) (r * 0.3 + g * 0.59 + b * 0.11);
                color[j * width + i] = grayInt * 1.0f / n;
                if (color[j * width + i] >= 0.5f) {
                    color[j * width + i] = 1.0f;
                } else {
                    color[j * width + i] = 0.0f;
                }
            }
        }
        return color;
    }

    public void loadImageDataToGrayFast(File file, float[] color, boolean normalization, boolean meanStd) throws Exception {
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
        int pixel = 0; // 下面三行代码将一个数字转换为RGB数字
        int r = 0;
        int g = 0;
        int b = 0;
        float n = 1.0f;
        int size = height * width;
        if (color.length != size) {
            throw new RuntimeException("color.length is not equals image size.");
        }
        if (normalization) {
            n = 255.0f;
        }
        for (int j = miny; j < height; j++) {
            for (int i = minx; i < width; i++) {
                pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字
                r = (pixel & 0xff0000) >> 16;
                g = (pixel & 0xff00) >> 8;
                b = (pixel & 0xff);
                int grayInt = (int) (r * 0.3 + g * 0.59 + b * 0.11);
                if (meanStd) {
                    color[i * height + j] = (float) ((grayInt * 1.0f / n) - mean[0]) / std[0];
                } else {
                    color[i * height + j] = grayInt * 1.0f / n;
                }
            }
        }
    }

    public BufferedImage convertRGBImage(int[][] rgbValue) {
        int height = rgbValue.length;
        int width = rgbValue[0].length;
        //		System.out.println(width + ":" + height);
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                bufferedImage.setRGB(x, y, rgbValue[y][x]);
            }
        }
        return bufferedImage;
    }

    public BufferedImage convertGrayImage(int[][] rgbValue) {
        int height = rgbValue.length;
        int width = rgbValue[0].length;
        //		System.out.println(width + ":" + height);
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                bufferedImage.setRGB(x, y, rgbValue[y][x]);
            }
        }
        return bufferedImage;
    }

    public boolean createGrayImage(String path, String extName, int[][] rgbValue) {
        BufferedImage bufferedImage = this.convertGrayImage(rgbValue);
        //		System.out.println(extName);
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

    public boolean createRGBImage(String path, String extName, int[][] rgb) {
        BufferedImage bufferedImage = this.convertRGBImage(rgb);
        //		System.out.println(extName);
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

    public boolean createRGBImage(String path, String extName, int[][] rgb, int[][] bbox) {
        BufferedImage bufferedImage = this.convertRGBImage(rgb);
        bufferedImage.getGraphics().setColor(Color.RED);
        for (int[] box : bbox) {
            int x = box[1] - box[3] / 2;
            int y = box[2] - box[4] / 2;
            bufferedImage.getGraphics().drawRect(x, y, box[3], box[4]);
        }
        //		System.out.println(extName);
        File outputfile = new File(path);
        //		System.out.println(path);
        try {
            System.out.println(ImageIO.write(bufferedImage, extName, outputfile));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean createRGBImage(String path, String extName, int[][] rgb, int weight, int height) {
        BufferedImage bufferedImage = this.convertRGBImage(rgb);
        File outputfile = new File(path);
        try {
            //			System.out.println(outputfile.exists());
            if (!outputfile.exists()) {
                BufferedImage output = resizeImage(bufferedImage, weight, height);
                ImageIO.write(output, extName, outputfile);
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean createRGBImage(String path, String extName, int[][] rgb, int weight, int height, int[][] bbox) {
        BufferedImage bufferedImage = this.convertRGBImage(rgb);
        //		System.out.println(extName);
        File outputfile = new File(path);
        try {
            BufferedImage output = resizeImage(bufferedImage, weight, height, bbox);
            ImageIO.write(output, extName, outputfile);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean createRGBImage(String path, String extName, int[][] rgb, int weight, int height, int[][] bbox, String[] classLabel) {
        BufferedImage bufferedImage = this.convertRGBImage(rgb);
        //		System.out.println(extName);
        File outputfile = new File(path);
        try {
            BufferedImage output = null;
            if (classLabel != null) {
                output = resizeImage(bufferedImage, weight, height, bbox, classLabel);
            } else {
                output = resizeImage(bufferedImage, weight, height, bbox);
            }
            ImageIO.write(output, extName, outputfile);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean createRGBGIF(String path, String extName, int[][][] rgb, int width, int height) {
        AnimatedGifEncoder ae = new AnimatedGifEncoder();
        ae.setSize(width, height);
        ae.start(path);
        ae.setDelay(1000);
        ae.setRepeat(0);
        for (int i = 0; i < rgb.length; i++) {
            BufferedImage bufferedImage = this.convertRGBImage(rgb[i]);
            ae.addFrame(bufferedImage);
        }
        return ae.finish();
    }

    public boolean createScaledRGBGIF(String path, String extName, int[][][] rgb, int targetWidth, int targetHeight) {
        AnimatedGifEncoder ae = new AnimatedGifEncoder();
        ae.setSize(targetWidth, targetHeight);
        ae.start(path);
        ae.setDelay(100);
        ae.setRepeat(0);
        for (int i = 0; i < rgb.length; i++) {
            BufferedImage bufferedImage = this.convertRGBImage(rgb[i]);
            Image resultingImage = bufferedImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
            BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
            outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
            ae.addFrame(outputImage);
        }
        return ae.finish();
    }

    /**
     * red 1 green 2 bule 3
     *
     * @param path
     * @param extName
     * @param rgb
     * @param rgbType
     * @return
     */
    public boolean createRGBImage(String path, String extName, float[][] rgb, int rgbType) {
        int[][] rgbInt = new int[rgb.length][rgb[0].length];
        for (int i = 0; i < rgb.length; i++) {
            for (int j = 0; j < rgb[i].length; j++) {
                if (rgbType == 1) { //red
                    rgbInt[i][j] = (int) rgb[i][j] << 16;
                } else if (rgbType == 2) {
                    rgbInt[i][j] = (int) rgb[i][j] << 8;
                } else {
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
     * red 1 green 2 bule 3
     *
     * @param path
     * @param extName
     * @param rgb
     * @param rgbType
     * @return
     */
    public boolean createRGBImage(String path, String extName, int height, int width, float[] rgb, int rgbType) {
        int[][] rgbInt = new int[height][width];
        for (int i = 0; i < rgb.length; i++) {
            for (int j = 0; j < width; j++) {
                if (rgbType == 1) { //red
                    rgbInt[i][j] = (int) rgb[i * width + j] << 16;
                } else if (rgbType == 2) {
                    rgbInt[i][j] = (int) rgb[i * width + j] << 8;
                } else {
                    rgbInt[i][j] = (int) rgb[i * width + j];
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

    public boolean createImage(int index, float[] data, String label, int height, int width, String filePath, String extName) {
        try {
            int[][] rgbInt = color2rgb(data, height, width);
            BufferedImage bufferedImage = this.convertRGBImage(rgbInt);
            System.out.println(extName);
            File outputfile = new File(filePath + "/" + index + "_" + label + "." + extName);
            try {
                System.out.println(ImageIO.write(bufferedImage, extName, outputfile));
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return false;
            }
            return true;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return false;
    }

    public static void createGifFromFolder(String inputFolderPath, String outputGifPath, int delay, boolean loop) {
        File inputFolder = new File(inputFolderPath);

        // 检查文件夹是否存在
        if (!inputFolder.exists() || !inputFolder.isDirectory()) {
            System.err.println("指定的文件夹不存在或不是一个目录: " + inputFolderPath);
            return;
        }

        // 获取文件夹中的所有图片文件
        File[] imageFiles = inputFolder.listFiles((dir, name) -> {
            String lowerName = name.toLowerCase();
            return lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg")
                    || lowerName.endsWith(".png") || lowerName.endsWith(".bmp");
        });

        // 检查是否有图片文件
        if (imageFiles == null || imageFiles.length == 0) {
            System.err.println("指定的文件夹中没有找到图片文件: " + inputFolderPath);
            return;
        }

        // 按文件名排序（假设文件名已经按顺序命名）
        Arrays.sort(imageFiles, Comparator.comparing(File::getName));

        // 创建GIF编码器
        AnimatedGifEncoder gifEncoder = new AnimatedGifEncoder();
        gifEncoder.start(outputGifPath);
        gifEncoder.setDelay(delay);  // 设置帧延迟（毫秒）
        gifEncoder.setRepeat(loop ? 0 : -1);  // 0=无限循环，-1=不循环

        try {
            // 添加每一帧到GIF
            for (File imageFile : imageFiles) {
                gifEncoder.addFrame(javax.imageio.ImageIO.read(imageFile));
            }
        } catch (IOException e) {
            System.err.println("处理图片时出错: " + e.getMessage());
            e.printStackTrace();
        } finally {
            gifEncoder.finish();
            System.out.println("GIF生成完成，保存到: " + outputGifPath);
        }
    }

    static class Interpolation {
        public double oneDimensionalBicubicInterpolation(double[] p, double x) {
            return p[1] + 0.5d * x * (p[2] - p[0] + x * (2.0d * p[0] - 5.0d * p[1] + 4.0d * p[2] - p[3] + x * (3.0d * (p[1] - p[2]) + p[3] - p[0])));
        }

        public double twoDimensionalBicubicInterpolation(double[][] p, double x, double y) {
            final double[] arr = new double[4];
            arr[0] = oneDimensionalBicubicInterpolation(p[0], y);
            arr[1] = oneDimensionalBicubicInterpolation(p[1], y);
            arr[2] = oneDimensionalBicubicInterpolation(p[2], y);
            arr[3] = oneDimensionalBicubicInterpolation(p[3], y);
            return oneDimensionalBicubicInterpolation(arr, x);
        }
    }
}

