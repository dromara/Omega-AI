//package com.omega.common.utils;
//
//import org.bytedeco.javacv.FFmpegFrameGrabber;
//import org.bytedeco.javacv.Frame;
//import org.bytedeco.javacv.FrameGrabber;
//import org.bytedeco.javacv.OpenCVFrameConverter;
//import org.bytedeco.opencv.global.opencv_imgproc;
//import org.bytedeco.opencv.opencv_core.Mat;
//
//import java.io.Closeable;
//import java.io.File;
//import java.io.IOException;
//import java.nio.ByteBuffer;
//
///**
// * 读取mp4文件
// * @author admin
// */
//public class VideoReader implements Closeable {
//    private final FFmpegFrameGrabber grabber;
//    private final OpenCVFrameConverter.ToMat toMat = new OpenCVFrameConverter.ToMat();
//
//    /**
//     * 输出的宽度
//     */
//    private final int targetWidth;
//    /**
//     * 输出的高度
//     */
//    private final int targetHeight;
//    /**
//     * 帧的步长
//     */
//    private final int frameStep;
//    /**
//     * 是否进行归一操作 将[0,255]  -->[0,1]
//     */
//    private final boolean normalize;
//    private final float scaleFactor;
//    /**
//     * 总帧数
//     */
//    private final int totalFrames;
//    /**
//     * 源宽度
//     */
//    private final int width;
//    /**
//     * 源高度
//     */
//    private final int height;
//    /**
//     * 源帧率
//     */
//    private final double frameRate;
//    /**
//     * 当前帧的下标
//     */
//    private int currentFrameIndex = 0;
//
//    // 复用 Mat，减少内存分配
//    private final Mat matRGB = new Mat();
//    private final Mat resized = new Mat();
//
//    public VideoReader(File videoFile) {
//        this(videoFile, 0, 0, 1, false);
//    }
//
//    public VideoReader(File videoFile, int frameStep, boolean normalize) {
//        this(videoFile, 0, 0, frameStep, normalize);
//    }
//
//    public VideoReader(File videoFile, int targetWidth, int targetHeight, int frameStep, boolean normalize) {
//        if (videoFile == null || !videoFile.exists()) {
//            throw new IllegalArgumentException("视频文件不存在: " + videoFile);
//        }
//        if (frameStep <= 0) {
//            throw new IllegalArgumentException("间隔步长必须 >= 1");
//        }
//
//        this.frameStep = frameStep;
//        this.normalize = normalize;
//        this.scaleFactor = normalize ? (1.0f / 255.0f) : 1.0f;
//
//        this.grabber = new FFmpegFrameGrabber(videoFile);
//        try {
//            this.grabber.start();
//        } catch (FrameGrabber.Exception e) {
//            throw new RuntimeException(e);
//        }
//
//        this.width = grabber.getImageWidth();
//        this.height = grabber.getImageHeight();
//        this.frameRate = grabber.getFrameRate() > 0 ? grabber.getFrameRate() : 25.0;
//        this.totalFrames = grabber.getLengthInFrames();
//
//        this.targetWidth = targetWidth > 0 ? targetWidth : this.width;
//        this.targetHeight = targetHeight > 0 ? targetHeight : this.height;
//    }
//
//    public float[][][][] nextBatch4D(int batchFrames) throws FrameGrabber.Exception {
//        if (batchFrames <= 0) throw new IllegalArgumentException("batchFrames 必须 > 0");
//
//        int T = 0;
//        float[][][][] batchData = new float[batchFrames][3][targetWidth][targetHeight];
//
//        while (T < batchFrames) {
//            Frame frame = advanceToNextKeptFrame();
//            if (frame == null) {
//                break;
//            }
//
//            Mat matBGR = toMat.convert(frame);
//            if (matBGR == null || matBGR.empty()) {
//                continue;
//            }
//
//            // BGR -> RGB
//            opencv_imgproc.cvtColor(matBGR, matRGB, opencv_imgproc.COLOR_BGR2RGB);
//
//            Mat usedMat = matRGB;
//            if (matRGB.cols() != targetWidth || matRGB.rows() != targetHeight) {
//                opencv_imgproc.resize(matRGB, resized, new org.bytedeco.opencv.opencv_core.Size(targetWidth, targetHeight));
//                usedMat = resized;
//            }
//
//            ByteBuffer buf = usedMat.createBuffer();
//            int W = usedMat.cols(), H = usedMat.rows();
//
//            for (int c = 0; c < 3; c++) {
//                for (int x = 0; x < W; x++) {
//                    for (int y = 0; y < H; y++) {
//                        int pos = (y * W + x) * 3 + c;
//                        batchData[T][c][x][y] = (buf.get(pos) & 0xFF) * scaleFactor;
//                    }
//                }
//            }
//            T++;
//        }
//
//        if (T < batchFrames) {
//            // 截断多余空间
//            float[][][][] truncated = new float[T][3][targetWidth][targetHeight];
//            System.arraycopy(batchData, 0, truncated, 0, T);
//            return truncated;
//        }
//
//        return batchData;
//    }
//
//
//    private Frame advanceToNextKeptFrame() throws FrameGrabber.Exception {
//        while (true) {
//            Frame f = grabber.grab();
//            if (f == null) {
//                return null;
//            }
//            currentFrameIndex++;
//            if (f.image != null && ((currentFrameIndex - 1) % frameStep == 0)) {
//                return f;
//            }
//        }
//    }
//
//    public int getTargetWidth() { return targetWidth; }
//    public int getTargetHeight() { return targetHeight; }
//    public double getFrameRate() { return frameRate; }
//    public int getTotalFrames() { return totalFrames; }
//    public FFmpegFrameGrabber getGrabber() { return grabber; }
//    public int getCurrentFrameIndex() { return currentFrameIndex; }
//
//    @Override
//    public void close() throws IOException {
//        try { grabber.stop(); } catch (Exception ignored) {}
//        try { grabber.close(); } catch (Exception ignored) {}
//    }
//}
