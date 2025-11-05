package com.omega.boot.starter.service;

import com.omega.boot.starter.entity.ModelData;
import com.omega.boot.starter.utils.FileUtils;
import com.omega.boot.starter.utils.JarUrlUtils;
import com.omega.boot.starter.utils.YoloUtils;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.yolo.data.DataType;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.model.YoloBox;
import com.omega.example.yolo.model.YoloDetection;
import com.omega.example.yolo.test.deepsort.*;
import com.omega.example.yolo.utils.LabelFileType;
import com.omega.example.yolo.utils.OMImage;
import jakarta.annotation.PostConstruct;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import org.bytedeco.ffmpeg.global.avcodec;
import org.springframework.util.StringUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.time.Duration;
import java.util.*;
import java.util.List;

/**
 * Yolo7模型初始化类
 *
 * @author haylee
 * @date 2025/09/06 14:33
 */
@Configuration
@ConditionalOnExpression("@modelConfig['yolov7'] != null")
public class Yolov7Service extends ModelAbstract{
    private Logger logger = LoggerFactory.getLogger(Yolov3Service.class);

    @Autowired
    @Qualifier("modelConfig")
    private Map<String, ModelData> modelConfig;

    private ModelData modelData;

    private static final String model_type = "yolov7";

    private int im_w = 416;
    private int im_h = 416;
    private int batchSize = 1;
    private int class_num = 2;

    private String[] labelset = new String[]{"unmask", "mask"};

    @Value("${model.cudnn:false}")
    private boolean cudnn;

    @PostConstruct
    public void init() {
        this.modelData = modelConfig.get(model_type);
    }

    @Bean("yolov7")
    public Yolo getNetwork() {
        try {
            String path = this.modelData.getPath();
            String cfg = this.modelData.getConfig().getStr("cfg");
            String name = this.modelData.getConfig().getStr("name");
            this.im_w = this.modelData.getConfig().getInt("image_width");
            this.im_h = this.modelData.getConfig().getInt("image_height");
            String labels = this.modelData.getConfig().getStr("labels");
            if(StringUtils.hasText( labels)){
                this.labelset = labels.split(",");
            }
            Yolo network = new Yolo(LossType.yolov7, UpdaterType.adamw);
            network.CUDNN = cudnn;
            network.RUN_MODEL = RunModel.TEST;
            ModelLoader.loadConfigToModel(network, path+ File.separator + cfg);
            network.init();
            ModelUtils.loadModel(network, path+ File.separator + name);
            return network;
        } catch (Exception e) {
            logger.error("Error loading yolo7: {}", e);
        }
        return null;
    }

    public String predict(String url) throws Exception {
        if (YoloUtils.yoloDetectImage( url)){
            return imageDetect(url);
        }
        if (YoloUtils.yoloDetectVideo( url)){
            return videoDetect(url);
        }
        return "";
    }

    /**
     * 图片检测
     * @param url
     * @return
     * @throws Exception
     */
    public String imageDetect(String url) throws Exception {
        File file = new File(url);
        Yolo netWork = getNetwork();
        DetectionDataLoader data = new DetectionDataLoader(file.getParent(), null, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
        List<YoloBox> draw_bbox = MBSGDOptimizer.showObjectRecognitionYoloV3(netWork, data, batchSize);
        String outputPath = FileUtils.mkdir(JarUrlUtils.getJarPath() + File.separator + "upload" + File.separator + model_type + File.separator + UUID.randomUUID().toString());
        List<String> list = showImg(outputPath, data, class_num, draw_bbox, batchSize, false, im_w, im_h, this.labelset);
        return list.size() > 0 ? list.get(0) : "";
    }

    /**
     * 视频跟踪检测
     * @param url
     * @return
     * @throws Exception
     */
    public String videoDetect(String url) throws Exception {

        FFmpegFrameGrabber grabber = null;
        File file = new File(url);
        try {
            grabber = new FFmpegFrameGrabber(file);
            grabber.setFrameRate(30);
            grabber.start();
        } catch (org.bytedeco.javacv.FFmpegFrameGrabber.Exception e) {
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

            BufferedImage bufferedImage = YoloUtils.toBufferedImage(frame,converter);

            List<Detection> detections = runYOLODetection(input, bufferedImage);

            List<Track> activeTracks = tracker.update(detections);

            BufferedImage newBufferedImage = YoloUtils.converter(bufferedImage, input.width, input.height, Color.BLACK);
            int sec = (int) (grabber.getTimestamp() / 1000f / 1000f);
            String formatTime = YoloUtils.formatTime(sec);

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
        return vedioUrl;
    }

    private List<Detection> runYOLODetection(Tensor input, BufferedImage image) {

        List<Detection> detections = new ArrayList<>();

        OMImage orig = ImageLoader.loadImage(image);

        ImageLoader.loadVailDataDetection(input, null, 0, orig, null, input.width, input.height, 0, 0);
        input.hostToDevice();

        Yolo netWork = getNetwork();
        Tensor[] output = netWork.predicts(input);

        List<BoundingBox> list = new ArrayList<BoundingBox>();

        for (int i = 0; i < netWork.outputLayers.size(); i++) {
            YoloLayer layer = (YoloLayer) netWork.outputLayers.get(i);
            YoloDetection[][] dets = com.omega.example.yolo.utils.YoloUtils.getYoloDetectionsV7(output[i], layer.anchors, layer.mask, layer.bbox_num,
                    layer.outputs, layer.class_number, netWork.getHeight(), netWork.getWidth(), 0.5f);
            for (int j = 0; j < dets.length; j++) {
                YoloDetection[] yoloDetections = dets[j];
                for (int k = 0; k < yoloDetections.length; k++) {
                    YoloDetection yoloDetection = yoloDetections[k];
                    if (yoloDetection.getObjectness() <= 0.7) {
                        continue;
                    }
                    int classes = (int) yoloDetection.getClasses();
                    BoundingBox boundingBox = new BoundingBox(yoloDetection.getBbox(), yoloDetection.getProb()[classes],
                            classes);
                    list.add(boundingBox);
                }
            }
        }

        List<BoundingBox> nms = BoundingBox.nms(list);

        for (BoundingBox boundingBox : nms) {

            int x = (int) boundingBox.getX(input.width);
            int y = (int) boundingBox.getY(input.height);
            int width = (int) boundingBox.getWidth(input.width);
            int height = (int) boundingBox.getHeight(input.height);

            Rect bbox = new Rect(x, y, width, height);

            detections.add(new Detection(bbox, boundingBox.confidence, labelset[boundingBox.predictedClass],
                    boundingBox.predictedClass));
        }

        return detections;
    }



    public String getModelPath(){
        return this.modelData.getPath();
    }
}
