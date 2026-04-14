package com.omega.example.opensora.utils;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;
import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;

/**
 * 视频读取器 - 支持读取视频帧并处理
 * 帧率设置为30FPS
 */
public class VideoReader {

    private final String videoPath;
    private FFmpegFrameGrabber grabber;
    private final Java2DFrameConverter frameConverter;
    private final int targetFPS;

    /**
     * 构造函数
     * @param videoPath 视频文件路径
     */
    public VideoReader(String videoPath) {
        this(videoPath, 30);
    }

    /**
     * 构造函数 - 自定义帧率
     * @param videoPath 视频文件路径
     * @param targetFPS 目标帧率
     */
    public VideoReader(String videoPath, int targetFPS) {
        this.videoPath = videoPath;
        this.targetFPS = targetFPS;
        this.frameConverter = new Java2DFrameConverter();
    }

    /**
     * 打开视频文件
     */
    public void open() throws FrameGrabber.Exception {
        grabber = new FFmpegFrameGrabber(videoPath);
        grabber.start();
        System.out.println("视频已打开: " + videoPath);
        System.out.println("原始帧率: " + grabber.getVideoFrameRate() + " FPS");
        System.out.println("视频分辨率: " + grabber.getImageWidth() + "x" + grabber.getImageHeight());
        System.out.println("视频时长: " + grabber.getLengthInTime() / 1000000 + " 秒");
        System.out.println("目标帧率: " + targetFPS + " FPS");
    }

    /**
     * 读取下一帧
     * @return BufferedImage对象，如果没有更多帧则返回null
     */
    public BufferedImage readNextFrame() throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        Frame frame = grabber.grab();
        if (frame == null || frame.image == null) {
            return null;
        }

        BufferedImage converted = frameConverter.convert(frame);
        return deepCopy(converted);
    }

    /**
     * 读取指定时间点的帧
     * @param timestamp 时间戳（毫秒）
     * @return BufferedImage对象
     */
    public BufferedImage readFrameAt(long timestamp) throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        // 设置时间戳（微秒）
        grabber.setTimestamp(timestamp * 1000);
        Frame frame = grabber.grab();
        if (frame == null || frame.image == null) {
            return null;
        }

        BufferedImage converted = frameConverter.convert(frame);
        return deepCopy(converted);
    }

    /**
     * 按固定帧率读取帧（跳帧处理）
     * @param frameIndex 帧索引
     * @return BufferedImage对象
     */
    public BufferedImage readFrameByFPS(int frameIndex) throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        // 计算目标时间戳
        double frameTime = 1000.0 / targetFPS; // 每帧的毫秒数
        long targetTimestamp = (long) (frameIndex * frameTime * 1000); // 转换为微秒

        grabber.setTimestamp(targetTimestamp);

        // 跳过非图像帧，确保定位准确
        Frame frame;
        do {
            frame = grabber.grab();
        } while (frame != null && frame.image == null);

        if (frame == null) {
            return null;
        }

        BufferedImage converted = frameConverter.convert(frame);
        return deepCopy(converted);
    }

    /**
     * 按固定帧率顺序读取所有帧（更可靠的方法）
     * @param maxFrames 最大帧数
     * @return 帧列表
     */
    public java.util.List<BufferedImage> readAllFramesByFPS(int maxFrames) throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        java.util.List<BufferedImage> frames = new java.util.ArrayList<>();

        // 重置到开头
        grabber.setTimestamp(0);

        double originalFrameRate = grabber.getVideoFrameRate();
        double skipRatio = originalFrameRate / targetFPS; // 需要跳过的帧数比例
        int skipCounter = 0;
        int targetSkip = (int) Math.round(skipRatio);

        System.out.println("原始帧率: " + originalFrameRate + ", 目标帧率: " + targetFPS + ", 跳帧比例: " + skipRatio);

        int framesRead = 0;

        while (framesRead < maxFrames) {
            Frame frame = grabber.grab();

            if (frame == null) {
                break; // 视频结束
            }
            if (frame.image != null) {
                if (skipCounter >= targetSkip || targetSkip == 0) {
                    BufferedImage converted = frameConverter.convert(frame);
                    // 创建深拷贝，避免所有引用指向同一个对象
                    BufferedImage copy = deepCopy(converted);
                    frames.add(copy);
                    framesRead++;
                    skipCounter = 0;
                } else {
                    skipCounter++;
                }
            }
        }

        return frames;
    }

    /**
     * 创建 BufferedImage 的深拷贝（转换为 RGB）
     * 这很重要，因为 Java2DFrameConverter 可能重用内部缓冲区
     */
    private static BufferedImage deepCopy(BufferedImage source) {
        if (source == null) {
            return null;
        }

        int width = source.getWidth();
        int height = source.getHeight();

        // 如果源是 BGR 类型，手动转换
        if (source.getType() == TYPE_3BYTE_BGR) {
            BufferedImage rgbCopy = new BufferedImage(width, height, TYPE_INT_RGB);
            byte[] bgrData = ((DataBufferByte) source.getRaster().getDataBuffer()).getData();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = (y * width + x) * 3;
                    int b = bgrData[index] & 0xFF;
                    int g = bgrData[index + 1] & 0xFF;
                    int r = bgrData[index + 2] & 0xFF;
                    int rgb = (r << 16) | (g << 8) | b;
                    rgbCopy.setRGB(x, y, rgb);
                }
            }
            return rgbCopy;
        }

        // 其他类型，使用 Graphics 拷贝
        BufferedImage copy = new BufferedImage(width, height, source.getType());
        Graphics g = copy.getGraphics();
        g.drawImage(source, 0, 0, null);
        g.dispose();
        return copy;
    }

    // ==================== Resize 功能 ====================

    /**
     * UCFCenterCropVideo 方式调整帧大小（正方形）
     * 算法：先将短边缩放到目标尺寸，然后从中心裁剪
     *
     * 示例：原图 1920x1080，目标 256x256
     * 1. 短边 1080 -> 256，缩放比例 = 256/1080
     * 2. 缩放后尺寸 ≈ 455x256
     * 3. 中心裁剪 256x256
     *
     * @param frame 原始帧
     * @param targetSize 目标尺寸（宽=高）
     * @return 调整大小后的帧
     */
    public static BufferedImage resizeCenterCrop(BufferedImage frame, int targetSize) {
        return resizeCenterCrop(frame, targetSize, targetSize);
    }

    /**
     * UCFCenterCropVideo 方式调整帧大小（自定义宽高）
     * 算法：先将短边缩放到目标尺寸的最小值，然后从中心裁剪
     *
     * 示例：原图 1920x1080，目标 320x240
     * 1. 短边 1080 -> 240，缩放比例 = 240/1080
     * 2. 缩放后尺寸 ≈ 427x240
     * 3. 中心裁剪 320x240
     *
     * @param frame 原始帧（RGB 或 BGR 格式均可）
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @return 调整大小后的帧（RGB 格式）
     */
    public static BufferedImage resizeCenterCrop(BufferedImage frame, int targetWidth, int targetHeight) {
        if (frame == null) {
            return null;
        }

        int width = frame.getWidth();
        int height = frame.getHeight();

        // 如果已经是目标尺寸，直接返回拷贝
        if (width == targetWidth && height == targetHeight) {
            return deepCopy(frame);
        }

        // 计算缩放比例（基于目标尺寸的最小值）
        int minTargetSize = Math.min(targetWidth, targetHeight);
        double scale = (double) minTargetSize / Math.min(width, height);
        int scaledWidth = (int) Math.round(width * scale);
        int scaledHeight = (int) Math.round(height * scale);

        // 先缩放到 RGB（Graphics.drawImage 会自动处理颜色转换）
        BufferedImage scaledImage = new BufferedImage(scaledWidth, scaledHeight, TYPE_INT_RGB);
        Graphics2D g2d = scaledImage.createGraphics();
        try {
            g2d.drawImage(frame, 0, 0, scaledWidth, scaledHeight, null);
        } finally {
            g2d.dispose();
        }

        // 然后从中心裁剪
        int x = (scaledWidth - targetWidth) / 2;
        int y = (scaledHeight - targetHeight) / 2;

        BufferedImage croppedImage = new BufferedImage(targetWidth, targetHeight, TYPE_INT_RGB);
        Graphics g = croppedImage.getGraphics();
        try {
            g.drawImage(scaledImage, 0, 0, targetWidth, targetHeight,
                    x, y, x + targetWidth, y + targetHeight, null);
        } finally {
            g.dispose();
        }

        return croppedImage;
    }

    /**
     * UCFCenterCropVideo 方式批量调整帧大小（正方形）
     * @param frames 原始帧列表
     * @param targetSize 目标尺寸（宽=高）
     * @return 调整大小后的帧列表
     */
    public static java.util.List<BufferedImage> resizeCenterCrop(java.util.List<BufferedImage> frames, int targetSize) {
        return resizeCenterCrop(frames, targetSize, targetSize);
    }

    /**
     * UCFCenterCropVideo 方式批量调整帧大小（自定义宽高）
     * @param frames 原始帧列表
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @return 调整大小后的帧列表
     */
    public static java.util.List<BufferedImage> resizeCenterCrop(java.util.List<BufferedImage> frames, int targetWidth, int targetHeight) {
        java.util.List<BufferedImage> resizedFrames = new java.util.ArrayList<>(frames.size());
        for (BufferedImage frame : frames) {
            resizedFrames.add(resizeCenterCrop(frame, targetWidth, targetHeight));
        }
        return resizedFrames;
    }

    /**
     * 简单缩放（输出 RGB 格式）
     * @param frame 原始帧
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @return 缩放后的帧
     */
    public static BufferedImage resize(BufferedImage frame, int targetWidth, int targetHeight) {
        if (frame == null) {
            return null;
        }

        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        try {
            g2d.drawImage(frame, 0, 0, targetWidth, targetHeight, null);
        } finally {
            g2d.dispose();
        }
        return resized;
    }

    /**
     * 简单缩放到正方形
     * @param frame 原始帧
     * @param targetSize 目标尺寸
     * @return 缩放后的帧
     */
    public static BufferedImage resize(BufferedImage frame, int targetSize) {
        return resize(frame, targetSize, targetSize);
    }

    /**
     * 批量简单缩放
     * @param frames 原始帧列表
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @return 缩放后的帧列表
     */
    public static java.util.List<BufferedImage> resize(java.util.List<BufferedImage> frames, int targetWidth, int targetHeight) {
        java.util.List<BufferedImage> resizedFrames = new java.util.ArrayList<>(frames.size());
        for (BufferedImage frame : frames) {
            resizedFrames.add(resize(frame, targetWidth, targetHeight));
        }
        return resizedFrames;
    }

    /**
     * 获取视频信息
     */
    public VideoInfo getVideoInfo() throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        VideoInfo info = new VideoInfo();
        info.width = grabber.getImageWidth();
        info.height = grabber.getImageHeight();
        info.frameRate = grabber.getVideoFrameRate();
        info.duration = grabber.getLengthInTime() / 1000000.0;
        info.totalFrames = (long) (info.frameRate * info.duration);

        return info;
    }

    /**
     * 跳转到指定帧位置
     */
    public void seekToFrame(long frameNumber) throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }

        double frameTime = 1000.0 / grabber.getVideoFrameRate();
        long timestamp = (long) (frameNumber * frameTime * 1000);
        grabber.setTimestamp(timestamp);
    }

    /**
     * 重置到视频开头
     */
    public void reset() throws FrameGrabber.Exception {
        if (grabber == null) {
            throw new IllegalStateException("视频未打开，请先调用open()方法");
        }
        grabber.setTimestamp(0);
    }

    /**
     * 关闭视频
     */
    public void close() throws FrameGrabber.Exception {
        if (grabber != null) {
            grabber.stop();
            grabber.release();
            grabber = null;
            System.out.println("视频已关闭");
        }
    }

    /**
     * 获取当前帧率
     */
    public int getTargetFPS() {
        return targetFPS;
    }

    /**
     * 视频信息类
     */
    public static class VideoInfo {
        public int width;
        public int height;
        public double frameRate;
        public double duration; // 秒
        public long totalFrames;

        @Override
        public String toString() {
            return String.format("VideoInfo[分辨率: %dx%d, 帧率: %.2f FPS, 时长: %.2f秒, 总帧数: %d]",
                    width, height, frameRate, duration, totalFrames);
        }
    }
}
