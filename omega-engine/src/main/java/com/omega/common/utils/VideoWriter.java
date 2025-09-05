package com.omega.common.utils;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

import static org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264;
import static org.opencv.core.CvType.CV_8UC3;

/**
 * 写入mp4文件  要在最后执行close 方法
 *
 * @author admin
 */
public class VideoWriter implements Closeable {
    /**
     * 是否进行了归一操作 [0,255]  -->[0,1]
     */
    private final boolean normalize;
    private final float scaleFactor;
    /**
     * 输出视频文件
     */
    private final File outFile;

    /**
     * 是否覆盖现有文件（true = 删除旧文件并重写）
     */
    private final boolean overwrite;

    /**
     * 输出视频宽度
     */
    private final int width;

    /**
     * 输出视频高度
     */
    private final int height;

    /**
     * 输出视频帧率
     */
    private final double fps;

    /**
     * 视频编码格式（例如 H.264）
     */
    private final int videoCodec;

    /**
     * 视频录制器
     */
    private FFmpegFrameRecorder recorder;

    /**
     * Mat 转换器（Mat -> Frame）
     */
    private final OpenCVFrameConverter.ToMat toMat = new OpenCVFrameConverter.ToMat();

    // 当使用 APPEND 且 outFile 已存在时，我们将写入到临时文件并在 close() 时替换。
    private File tempFileForAppend;
    private FFmpegFrameGrabber appendReader;

    // 复用的 Mat & 缓冲区
    private final Mat matRGB;
    private final Mat matBGR;
    private final ByteBuffer bufferRGB;
    private final ByteBuffer bufferBGR;


    public VideoWriter(File outFile, boolean overwrite, int width, int height, double fps, boolean normalize) throws Exception {
        this(outFile, overwrite, width, height, fps, AV_CODEC_ID_H264, normalize);
    }

    public VideoWriter(File outFile, boolean overwrite, int width, int height, double fps, int videoCodec, boolean normalize) throws Exception {
        Objects.requireNonNull(outFile, "outFile");
        this.outFile = outFile;
        this.overwrite = overwrite;
        this.width = width;
        this.height = height;
        this.normalize = normalize;
        this.scaleFactor = normalize ? 255.0f : 1.0f;

        this.fps = fps <= 0 ? 25.0 : fps;
        this.videoCodec = videoCodec;
        // 初始化 BGR Mat（OpenCV 默认格式）
        this.matRGB = new Mat(height, width, CV_8UC3);
        this.matBGR = new Mat(height, width, CV_8UC3);
        this.bufferRGB = matRGB.createBuffer();
        this.bufferBGR = matBGR.createBuffer();
        if (!this.overwrite) {
            // 读取已有内容，准备在 close() 时合并
            if (!outFile.exists()) {
                outFile.createNewFile();
            }

            this.tempFileForAppend = new File(outFile.getParentFile(), outFile.getName() + ".appending.tmp.mp4");
            if (tempFileForAppend.exists()) {
                tempFileForAppend.delete();
            }
            initRecorder(tempFileForAppend);
            // 先把旧视频复制一遍到 recorder
            appendReader = new FFmpegFrameGrabber(outFile);
            appendReader.start();
            Frame f;
            while ((f = appendReader.grabImage()) != null) {
                recorder.record(f);
            }
        } else {
            // 直接覆盖写入
            if (outFile.exists()) {
                outFile.delete();
            }
            initRecorder(outFile);
        }
    }

    private void initRecorder(File file) throws FrameRecorder.Exception {
        recorder = new FFmpegFrameRecorder(file, width, height);
        recorder.setFrameRate(fps);
        recorder.setFormat("mp4");
        recorder.setVideoCodec(videoCodec);
//        recorder.setPixelFormat(avcodec.AV_CODEC_ID_YUV4);
        recorder.start();
    }

    public void writeBatch(float[][][][] data) throws FrameRecorder.Exception {
        writeBatch(data, null, null);
    }

    /**
     * 写入一个 4D 张量批次，布局 (T,C,W,H)，C 必须为 3（RGB）。
     */
    public void writeBatch(float[][][][] data, float[] mean, float[] std) throws FrameRecorder.Exception {
        int T = data.length;
        if (T == 0) {
            return;
        }
        int C = data[0].length;
        int W = data[0][0].length;
        int H = data[0][0][0].length;

        if (C != 3) {
            throw new IllegalArgumentException("颜色必须为RGB 3通道");
        }
        if (W != width || H != height) {
            throw new IllegalArgumentException("输出尺寸与数据不符合");
        }

        for (int t = 0; t < T; t++) {
            bufferRGB.clear();
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    for (int c = 0; c < 3; c++) {
                        float v = data[t][c][x][y] * scaleFactor;
                        if (null != mean) {
                            v = (data[t][c][x][y] * std[c] + mean[c]) * scaleFactor;
                        }
                        int iv = Math.max(0, Math.min(255, Math.round(v)));
                        bufferRGB.put((byte) (iv & 0xFF));
                    }
                }
            }

            // 转换为 BGR 适配 OpenCV
            opencv_imgproc.cvtColor(matRGB, matBGR, opencv_imgproc.COLOR_RGB2BGR);
            bufferRGB.rewind();
            Frame frame = toMat.convert(matBGR);
            recorder.record(frame);
        }
    }

    @Override
    public void close() throws IOException {
        try {
            if (appendReader != null) {
                try {
                    appendReader.stop();
                } catch (Exception ignore) {
                }
                try {
                    appendReader.close();
                } catch (Exception ignore) {
                }
            }
            if (recorder != null) {
                try {
                    recorder.stop();
                } catch (Exception ignore) {
                }
                try {
                    recorder.close();
                } catch (Exception ignore) {
                }

            }
            if (tempFileForAppend != null) {
                // 将临时文件替换原文件
                if (outFile.exists() && !outFile.delete()) {
                    throw new IOException("无法删除要追加的原始文件： " + outFile);
                }
                if (!tempFileForAppend.renameTo(outFile)) {
                    throw new IOException("重命名文件失败： " + tempFileForAppend + " -> " + outFile);
                }
            }
        } finally {
            appendReader = null;
            recorder = null;
        }
    }
}
