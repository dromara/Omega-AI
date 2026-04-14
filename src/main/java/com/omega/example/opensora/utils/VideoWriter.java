package com.omega.example.opensora.utils;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.List;

import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;
import static java.awt.image.BufferedImage.TYPE_INT_RGB;

/**
 * 视频写入器 - 将 BufferedImage 数组转换为视频文件
 * 支持设置帧率、分辨率、码率等参数
 */
public class VideoWriter {

    private final String outputPath;
    private FFmpegFrameRecorder recorder;
    private final Java2DFrameConverter frameConverter;
    private final int fps;
    private final int width;
    private final int height;

    /**
     * 构造函数 - 使用默认参数
     * @param outputPath 输出视频路径
     * @param fps 输出帧率
     * @param width 视频宽度
     * @param height 视频高度
     */
    public VideoWriter(String outputPath, int fps, int width, int height) {
        this(outputPath, fps, width, height, 3000000);
    }

    /**
     * 构造函数 - 自定义码率
     * @param outputPath 输出视频路径
     * @param fps 输出帧率
     * @param width 视频宽度
     * @param height 视频高度
     * @param videoBitrate 视频码率（默认 3000000 = 3Mbps）
     */
    public VideoWriter(String outputPath, int fps, int width, int height, int videoBitrate) {
        this.outputPath = outputPath;
        this.fps = fps;
        this.width = width;
        this.height = height;
        this.frameConverter = new Java2DFrameConverter();

        try {
            recorder = new FFmpegFrameRecorder(outputPath, width, height);
            recorder.setVideoCodec(org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264);
            recorder.setFormat("mp4");
            recorder.setFrameRate(fps);
            recorder.setVideoBitrate(videoBitrate);
            // 移除 setPixelFormat，让 JavaCV 自动处理
            // recorder.setPixelFormat(org.bytedeco.ffmpeg.global.avutil.AV_PIX_FMT_YUV420P);
        } catch (Exception e) {
            throw new RuntimeException("初始化录制器失败", e);
        }
    }

    /**
     * 从 VideoReader 创建 VideoWriter（自动匹配参数）
     */
    public static VideoWriter fromVideoReader(String outputPath, VideoReader reader) throws Exception {
        VideoReader.VideoInfo info = reader.getVideoInfo();
        return new VideoWriter(outputPath, reader.getTargetFPS(), info.width, info.height);
    }

    /**
     * 打开/创建输出视频文件
     */
    public void open() throws Exception {
        if (recorder != null) {
            recorder.start();
            System.out.println("视频录制器已启动: " + outputPath);
            System.out.println("帧率: " + fps + " FPS");
            System.out.println("分辨率: " + width + "x" + height);
        }
    }

    /**
     * 写入单帧
     * @param frame 图像帧（RGB 或 BGR 格式均可）
     */
    public void writeFrame(BufferedImage frame) throws Exception {
        if (recorder == null) {
            throw new IllegalStateException("录制器未启动，请先调用 open() 方法");
        }

        // 如果是 RGB 格式，转换为 BGR
        BufferedImage frameToWrite = frame;
        if (frame.getType() == TYPE_INT_RGB) {
            frameToWrite = convertRGBToBGR(frame);
        }

        Frame cvFrame = frameConverter.convert(frameToWrite);
        if (cvFrame != null) {
            recorder.record(cvFrame);
        }
    }

    /**
     * 将 RGB 格式的 BufferedImage 转换为 BGR 格式
     * 手动转换确保颜色正确
     */
    private static BufferedImage convertRGBToBGR(BufferedImage rgbImage) {
        if (rgbImage == null) {
            return null;
        }

        int width = rgbImage.getWidth();
        int height = rgbImage.getHeight();

        // 创建 BGR 图像
        BufferedImage bgrImage = new BufferedImage(width, height, TYPE_3BYTE_BGR);
        byte[] bgrData = ((DataBufferByte) bgrImage.getRaster().getDataBuffer()).getData();

        // 手动转换 RGB 到 BGR
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = rgbImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int index = (y * width + x) * 3;
                bgrData[index] = (byte) b;
                bgrData[index + 1] = (byte) g;
                bgrData[index + 2] = (byte) r;
            }
        }

        return bgrImage;
    }

    /**
     * 写入多帧
     * @param frames 图像帧数组
     */
    public void writeFrames(BufferedImage[] frames) throws Exception {
        for (BufferedImage frame : frames) {
            writeFrame(frame);
        }
    }

    /**
     * 写入多帧（List）
     * @param frames 图像帧列表
     */
    public void writeFrames(List<BufferedImage> frames) throws Exception {
        for (BufferedImage frame : frames) {
            writeFrame(frame);
        }
    }

    /**
     * 快速写入 - 一站式写入所有帧
     * @param frames 图像帧数组
     */
    public void writeAll(BufferedImage[] frames) throws Exception {
        open();
        for (int i = 0; i < frames.length; i++) {
            writeFrame(frames[i]);
            if ((i + 1) % 30 == 0) {
                System.out.printf("已写入 %d/%d 帧%n", i + 1, frames.length);
            }
        }
        close();
    }

    /**
     * 快速写入 - 一站式写入所有帧（List）
     * @param frames 图像帧列表
     */
    public void writeAll(List<BufferedImage> frames) throws Exception {
        open();
        System.out.println("开始写入 " + frames.size() + " 帧到视频...");
        for (int i = 0; i < frames.size(); i++) {
            BufferedImage frame = frames.get(i);
            if (frame != null) {
                writeFrame(frame);
                if ((i + 1) % 5 == 0 || i == frames.size() - 1) {
                    System.out.printf("已写入 %d/%d 帧 (尺寸: %dx%d)%n", i + 1, frames.size(), frame.getWidth(), frame.getHeight());
                }
            } else {
                System.err.printf("警告: 第 %d 帧为 null，跳过%n", i);
            }
        }
        close();
    }

    /**
     * 设置音频参数（如果需要音频）
     */
    public void setAudioParams(int sampleRate, int channels, int audioBitrate) {
        if (recorder != null) {
            recorder.setAudioChannels(channels);
            recorder.setSampleRate(sampleRate);
            recorder.setAudioBitrate(audioBitrate);
        }
    }

    /**
     * 设置视频编码质量（1-51，值越小质量越高）
     */
    public void setQuality(int crf) {
        if (recorder != null) {
            recorder.setVideoOption("crf", String.valueOf(crf));
        }
    }

    /**
     * 设置预设编码速度
     * @param preset ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
     */
    public void setPreset(String preset) {
        if (recorder != null) {
            recorder.setVideoOption("preset", preset);
        }
    }

    /**
     * 获取输出路径
     */
    public String getOutputPath() {
        return outputPath;
    }

    /**
     * 获取帧率
     */
    public int getFPS() {
        return fps;
    }

    /**
     * 关闭录制器并完成视频写入
     */
    public void close() throws Exception {
        if (recorder != null) {
            recorder.stop();
            recorder.release();
            recorder = null;
            System.out.println("视频已保存: " + outputPath);
        }
    }

    /**
     * 释放资源
     */
    public void release() {
        if (frameConverter != null) {
            frameConverter.close();
        }
    }
}
