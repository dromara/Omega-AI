package com.omega.boot.starter.utils;

import com.omega.example.yolo.test.deepsort.LetterBox;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.*;
import java.nio.ByteBuffer;
import java.time.Duration;

public class YoloUtils {

    public static final String[] imageExtensions = {".png",".jpg"};
    public static final String[] vedioExtensions = {".mp4","avi"};

    public static boolean yoloDetectImage(String fileName) {
        String lowerFileName = fileName.toLowerCase();
        for (String ext : imageExtensions) {
            if (lowerFileName.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    public static boolean yoloDetectVideo(String fileName) {
        String lowerFileName = fileName.toLowerCase();
        for (String ext : vedioExtensions) {
            if (lowerFileName.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    public static String formatTime(int seconds) {
        Duration duration = Duration.ofSeconds(seconds);
        long hours = duration.toHours();
        long minutes = duration.toMinutes();
        long remainingSeconds = duration.getSeconds() % 60;
        return String.format("%02d:%02d", minutes, remainingSeconds);
    }

    public static BufferedImage toBufferedImage(Frame frame, Java2DFrameConverter converter) {
        // 使用 JavaCV 自带的转换器，它通常能正确处理颜色空间
        try {
            BufferedImage image = converter.convert(frame);
            if (image != null) {
                return image;
            }
        } catch (Exception e) {
            // 如果转换失败，回退到原始方法
        }

        // 回退到原始方法，但使用正确的通道顺序
        ByteBuffer buffer = (ByteBuffer) frame.image[0].position(0);
        byte[] framePixels = new byte[buffer.limit()];
        buffer.get(framePixels);

        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ColorModel cm = new ComponentColorModel(cs, false, false, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);

        // 尝试不同的通道顺序
        WritableRaster wr = Raster.createWritableRaster(
                new ComponentSampleModel(DataBuffer.TYPE_BYTE,
                        frame.imageWidth, frame.imageHeight, frame.imageChannels,
                        frame.imageStride, new int[] {2, 1, 0}), // BGR to RGB
                null);

        byte[] bufferPixels = ((DataBufferByte) wr.getDataBuffer()).getData();
        System.arraycopy(framePixels, 0, bufferPixels, 0, framePixels.length);

        return new BufferedImage(cm, wr, false, null);
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
    public static BufferedImage converter(BufferedImage bufferedImage, int width, int height, Color backColor) {
        int imgWidth = width;
        int imgHeight = height;

        // 使用原始图像的类型，而不是强制转换为 TYPE_INT_RGB
        int imageType = bufferedImage.getType();
        if (imageType == BufferedImage.TYPE_CUSTOM) {
            imageType = BufferedImage.TYPE_INT_RGB; // 回退方案
        }

        BufferedImage tempImg = new BufferedImage(width, height, imageType);
        Graphics2D graphics2D = tempImg.createGraphics();

        // 设置高质量的渲染参数
        graphics2D.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        graphics2D.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_QUALITY);

        // 设置背景色
        if (backColor != null) {
            graphics2D.setColor(backColor);
            graphics2D.fillRect(0, 0, width, height);
        } else {
            // 如果没有指定背景色，使用黑色
            graphics2D.setColor(Color.BLACK);
            graphics2D.fillRect(0, 0, width, height);
        }

        // 计算缩放尺寸
        double constrainRatio = (double) width / (double) height;
        double imageRatio = (double) bufferedImage.getWidth() / (double) bufferedImage.getHeight();
        if (constrainRatio < imageRatio) {
            imgHeight = (int) (width / imageRatio);
        } else {
            imgWidth = (int) (height * imageRatio);
        }

        int dx = (width - imgWidth) / 2;
        int dy = (height - imgHeight) / 2;

        // 高质量绘制图像
        graphics2D.drawImage(bufferedImage, dx, dy, imgWidth, imgHeight, null);
        graphics2D.dispose();

        return tempImg;
    }
}
