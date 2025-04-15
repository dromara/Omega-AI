package com.omega.example.yolo.test;

import com.omega.common.config.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.yolo.data.DataType;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.data.YoloDataTransform2;
import com.omega.example.yolo.model.YoloBox;
import com.omega.example.yolo.model.YoloDetection;
import com.omega.example.yolo.utils.*;

import java.util.ArrayList;
import java.util.List;

public class YoloV2Test {
    public static void showImg(String outputPath, DataSet dataSet, int class_num, List<YoloBox> score_bbox, int batchSize, boolean format, int im_w, int im_h) {
        ImageUtils utils = new ImageUtils();
        int lastIndex = dataSet.number % batchSize;
        for (int b = 0; b < dataSet.number; b++) {
            float[] once = dataSet.getOnceData(b).data;
            int bbox_index = b;
            if (b >= (dataSet.number - lastIndex)) {
                bbox_index = b + (batchSize - lastIndex);
            }
            YoloBox box = score_bbox.get(bbox_index);
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < box.getDets().size(); l++) {
                if (box.getDets().get(l) != null && box.getDets().get(l).getObjectness() > 0) {
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                YoloDetection det = box.getDets().get(index);
                bbox[i][0] = 0;
                bbox[i][1] = (int) ((det.getBbox()[0] - det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][2] = (int) ((det.getBbox()[1] - det.getBbox()[3] / 2.0f) * im_h);
                bbox[i][3] = (int) ((det.getBbox()[0] + det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][4] = (int) ((det.getBbox()[1] + det.getBbox()[3] / 2.0f) * im_h);
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox);
        }
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            YoloV2Test y = new YoloV2Test();
            //			y.yolov2_tiny_voc();
            y.yolov2_tiny();
            //			y.getAnchors();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void yolov3() {
    }

    public void yolov2_tiny() {
        int im_w = 256;
        int im_h = 256;
        int batchSize = 64;
        int class_num = 1;
        try {
            String cfg_path = "H:\\voc\\banana-detection\\yolov2-tiny-banana.cfg";
            String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
            String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
            String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
            String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
            YoloDataLoader trainData = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, im_w, im_h, 5, LabelType.csv_v3, true);
            YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, im_w, im_h, 5, LabelType.csv_v3, true);
            DataSet trainSet = YoloLabelUtils.formatToYoloV3(trainData.getDataSet(), im_w, im_h);
            DataSet vailSet = YoloLabelUtils.formatToYoloV3(vailData.getDataSet(), im_w, im_h);
            Yolo netWork = new Yolo(LossType.yolov2, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.01f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.trainObjectRecognitionOutputs(trainSet, vailSet, false);
            //			/**
            //			 * 处理测试预测结果
            //			 */
            //			List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData.getDataSet(), batchSize);
            //
            //			YoloDataLoader testData = new YoloDataLoader(testPath, testLabelPath, 101, 3, im_w, im_h, 5, LabelType.csv_v3, false);
            //
            //			String outputPath = "H:\\voc\\banana-detection\\test_yolov3\\";
            //
            //			showImg(outputPath, testData.getDataSet(), 1, draw_bbox, batchSize, false, im_w, im_h);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void yolov2_tiny_voc() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 20;
        try {
            String cfg_path = "H:/voc/train/yolov3-tiny-voc.cfg";
            String trainPath = "H:\\voc\\train\\imgs";
            String trainLabelPath = "H:\\voc\\train\\labels\\yolov3.txt";
            String testPath = "H:\\voc\\test\\imgs";
            String testLabelPath = "H:\\voc\\test\\labels\\yolov3.txt";
            YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov2, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.0001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 3000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void getAnchors() {
        try {
            String trainLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
            Tensor box = new Tensor(100, 1, 1, 4);
            LabelUtils.loadBoxCSV(trainLabelPath, box);
            Tensor anchors = AnchorBoxUtils.getAnchorBox(box, 6);
            System.out.println(JsonUtils.toJson(anchors.data));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

