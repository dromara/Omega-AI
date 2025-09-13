package com.omega.boot.starter.service;

import com.omega.boot.starter.entity.ModelData;
import com.omega.boot.starter.utils.FileUtils;
import com.omega.boot.starter.utils.JarUrlUtils;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.yolo.data.DataType;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.model.YoloBox;
import com.omega.example.yolo.utils.LabelFileType;
import com.omega.example.yolo.utils.YoloImageUtils;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Yolo3模型初始化类
 *
 * @author haylee
 * @date 2025/09/06 14:33
 */
@Configuration
@ConditionalOnExpression("@modelConfig['yolov3'] != null")
public class Yolov3Service extends ModelAbstract{

    private Logger logger = LoggerFactory.getLogger(Yolov3Service.class);

    @Autowired
    @Qualifier("modelConfig")
    private Map<String, ModelData> modelConfig;

    private ModelData modelData;

    private static final String model_type = "yolov3";

    private int im_w = 416;
    private int im_h = 416;
    private int batchSize = 1;
    private int class_num = 2;

    @Value("${model.cudnn:false}")
    private boolean cudnn;
    @PostConstruct
    public void init() {
        this.modelData = modelConfig.get(model_type);
    }

    @Bean("yolov3")
    public Yolo getNetwork() {
        try {
            String path = this.modelData.getPath();
            String cfg = this.modelData.getConfig().getStr("cfg");
            String name = this.modelData.getConfig().getStr("name");
            this.im_w = this.modelData.getConfig().getInt("image_width");
            this.im_h = this.modelData.getConfig().getInt("image_height");
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = cudnn;
            netWork.RUN_MODEL = RunModel.TEST;
            ModelLoader.loadConfigToModel(netWork, path+ File.separator + cfg);
            ModelUtils.loadModel(netWork, path+ File.separator + name);
            return netWork;
        } catch (Exception e) {
            logger.error("Error loading yolov3: {}", e);
        }
        return null;
    }

    public String predict(String url) throws Exception {
        Yolo netWork = getNetwork();
        DetectionDataLoader data = new DetectionDataLoader(url, null, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(data, batchSize);
        String[] labelset = new String[]{"unmask", "mask"};
        String outputPath = FileUtils.mkdir(JarUrlUtils.getJarPath() + File.separator + model_type + File.separator + UUID.randomUUID().toString());
        List<String> list = showImg(outputPath, data, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
        return list.size() > 0 ? list.get(0) : "";
    }

    public String getModelPath(){
        return this.modelData.getPath();
    }
}
