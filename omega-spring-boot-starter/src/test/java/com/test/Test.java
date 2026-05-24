package com.test;

import com.omega.boot.starter.utils.YoloUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.test.deepsort.*;
import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

public class Test {

    private static final Logger logger = LoggerFactory.getLogger(Test.class);
    public static void main(String[] args) throws Exception {
        String url = "D:\\下载\\car_detect.mp4";
        FFmpegFrameGrabber grabber = null;
        File file = new File(url);
        try {
            grabber = new FFmpegFrameGrabber(file);
            grabber.setFrameRate(30);
            grabber.start();
        } catch (FFmpegFrameGrabber.Exception e) {
            logger.error("Error loading video: {}", e);
        }
        Frame grabImage = grabber.grabImage();
        if (null == grabber || null == grabImage) {
            throw new Exception("Unsupported Decoding Format");
        }
        int width = 416;
        int height = 416;
        Tensor input = new Tensor(1, 3, height, width, true);
        SORTTracker tracker = new SORTTracker();
        Frame frame = null;
        String vedioUrl = url.replace(file.getName(), UUID.randomUUID().toString() + ".mp4");
        // 创建视频记录器
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(vedioUrl, width, height);
        // 使用原始视频的参数
        recorder.setFormat("mp4");
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        recorder.setFrameRate(grabber.getFrameRate());
        recorder.setPixelFormat(avutil.AV_PIX_FMT_YUV420P);
        recorder.setVideoBitrate(grabber.getVideoBitrate());

        // 高质量编码参数
        recorder.setVideoOption("preset", "slow");
        recorder.setVideoOption("crf", "18");
        recorder.setVideoOption("tune", "film");
        recorder.start();
        Java2DFrameConverter converter = new Java2DFrameConverter();
        while (true) {
            // 由于是本地视频，并且需要看视频，按照帧率处理
            frame = grabber.grabAtFrameRate();

            if (null == frame) {
                break;
            }

            if (frame.type != Frame.Type.VIDEO || null == frame.image) {
                frame.close();
                continue;
            }

            BufferedImage bufferedImage = com.omega.boot.starter.utils.YoloUtils.toBufferedImage(frame,converter);


            BufferedImage newBufferedImage = YoloUtils.converter(bufferedImage, input.width, input.height, Color.BLACK);
            int sec = (int) (grabber.getTimestamp() / 1000f / 1000f);
            String formatTime = YoloUtils.formatTime(sec);
            List<Track> activeTracks = new ArrayList<>();
            activeTracks.add(new Track(1, new Rect(10, 10, 416, 416), "car", null));
            Set<Integer> set = new HashSet<>();
            for (Track track : activeTracks) {
                set.add(track.trackId);
            }

            for (Track track : activeTracks) {
                ImageTools.drawRect(newBufferedImage, track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height,
                        Color.RED);
                ImageTools.drawText(newBufferedImage, track.trackId + "_" + track.label, track.bbox.x, track.bbox.y,
                        Color.RED);
                ImageTools.drawText(newBufferedImage, "时间" + formatTime, 20, 20, Color.RED);
                //由于帧率设置为30，SORTTracker maxAge也为30，这里是初步估计的结果
                ImageTools.drawText(newBufferedImage, "每秒" + set.size() + "辆", 150, 20, Color.RED);
            }
            Frame newFrame = converter.convert(newBufferedImage);
            recorder.record(newFrame);
        }
        grabber.close();
        recorder.stop();
        recorder.release();
        logger.info("视频已保存到：{}", vedioUrl);
    }
}
