//package com.omega.example.video;
//
//
//import com.omega.common.utils.VideoReader;
//import com.omega.common.utils.VideoWriter;
//import org.bytedeco.ffmpeg.global.avcodec;
//import org.bytedeco.ffmpeg.global.avutil;
//import org.bytedeco.javacpp.Loader;
//
//import java.io.File;
//
//public class VideoTest {
//    public static void main(String[] args) {
//        File in = new File("D:\\big_buck_bunny2.mp4");
//        int resizeW = 320, resizeH = 240; // 设为原始大小可传 0
//        int step = 1; // 每 5 帧取 1 帧
//        Loader.load(avutil.class);
//        Loader.load(avcodec.class);
//
//        // 设置日志级别为QUIET
//        avutil.av_log_set_level(avutil.AV_LOG_ERROR);
//
//        //不在try 里面声明对象  需要在代码后面手动执行close
//        try (VideoReader reader = new VideoReader(in, 0, 0, step, true)) {
//            System.out.println("Video info: " + reader.getFrameRate());
//            System.out.println("Video height: " + reader.getTargetHeight());
//            System.out.println("Video width: " + reader.getTargetWidth());
//            System.out.println("Video totalFrames:" + reader.getTotalFrames());
//
//            // 覆盖写入输出
//            File outOverwrite = new File("d:\\out_overwrite.mp4");
//            VideoWriter writer = new VideoWriter(outOverwrite, true, reader.getTargetWidth(), reader.getTargetHeight(), reader.getFrameRate(), true);
//
////
//            while (true) {
//                float[][][][] floats = reader.nextBatch4D(12);
//                if (floats.length == 0) {
//                    break;
//                }
//                System.out.println(floats.length);
//                writer.writeBatch(floats);
//
//            }
//            //一定要执行close  否则 使用try 声明
//            writer.close();
//
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
//
//
//}