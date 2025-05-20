package com.omega.example.yolo.test;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.data.ImageData;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.yolo.data.DataType;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.data.YoloDataTransform2;
import com.omega.example.yolo.utils.*;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Administrator
 */
public class YoloV1Test {
    public static final String[] GL_CLASSES = new String[]{"person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"};
    public static int im_w = 256;
    public static int im_h = 256;

    public static void showImg(String outputPath, DataSet dataSet, int class_num, float[][][] score_bbox, boolean format) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < dataSet.number; b++) {
            float[] once = dataSet.getOnceData(b).data;
            float[][] label = score_bbox[b];
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < label.length; l++) {
                for (int c = 0; c < class_num; c++) {
                    if (label[l][c] == 0.0f) {
                        continue;
                    }
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                bbox[i][0] = 0;
                bbox[i][1] = (int) (label[index][class_num + 1] - label[index][class_num + 3] / 2);
                bbox[i][2] = (int) (label[index][class_num + 2] - label[index][class_num + 4] / 2);
                bbox[i][3] = (int) (label[index][class_num + 1] + label[index][class_num + 3] / 2);
                bbox[i][4] = (int) (label[index][class_num + 2] + label[index][class_num + 4] / 2);
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox);
        }
    }

    public static void showImg(String outputPath, YoloDataLoader dataLoader, int class_num, float[][][] score_bbox, boolean format) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < dataLoader.number; b++) {
            float[] once = dataLoader.loadData(b);
            float[][] label = score_bbox[b];
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < label.length; l++) {
                for (int c = 0; c < class_num; c++) {
                    if (label[l][c] == 0.0f) {
                        continue;
                    }
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                bbox[i][0] = getClass(label[index]);
                bbox[i][1] = (int) (label[index][class_num + 1] - label[index][class_num + 3] / 2);
                bbox[i][2] = (int) (label[index][class_num + 2] - label[index][class_num + 4] / 2);
                bbox[i][3] = (int) (label[index][class_num + 1] + label[index][class_num + 3] / 2);
                bbox[i][4] = (int) (label[index][class_num + 2] + label[index][class_num + 4] / 2);
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox);
        }
    }

    public static void showImg(String outputPath, YoloDataLoader dataLoader, int class_num, float[][][] score_bbox, int im_w, int im_h, boolean format) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < dataLoader.number; b++) {
            float[] once = dataLoader.loadData(b);
            float[][] label = score_bbox[b];
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < label.length; l++) {
                for (int c = 0; c < class_num; c++) {
                    if (label[l][c] == 0.0f) {
                        continue;
                    }
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                bbox[i][0] = getClass(label[index]);
                bbox[i][1] = (int) (label[index][class_num + 1] - label[index][class_num + 3] / 2);
                bbox[i][2] = (int) (label[index][class_num + 2] - label[index][class_num + 4] / 2);
                bbox[i][3] = (int) (label[index][class_num + 1] + label[index][class_num + 3] / 2);
                bbox[i][4] = (int) (label[index][class_num + 2] + label[index][class_num + 4] / 2);
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox, GL_CLASSES);
        }
    }

    public static int getClass(float[] bbox) {
        int class_index = 0;
        int class_pro = 0;
        for (int i = 0; i < bbox.length - 4; i++) {
            if (class_pro >= bbox[i]) {
                class_index = i;
            }
        }
        return class_index;
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            YoloV1Test t = new YoloV1Test();
            t.yolov1();
            //			t.showImg2();
            //			t.yolov1_tiny_voc();
            //			t.testTransforms();
            //			t.testYoloBBox();
            //			t.yolov1_show();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void showImg2() {
        try {
            int im_w = 1024;
            int im_h = 483;
            int img_size = 416;
            String trainPath = "H:\\voc\\helmet_dataset\\JPEGImages\\02049.jpg";
            String outputPath = "H:\\voc\\helmet_dataset\\00000.jpg";
            String outputPath2 = "H:\\voc\\helmet_dataset\\02049a.jpg";
            File file = new File(trainPath);
            ImageData data = YoloImageUtils.IU().getImageData(file);
            //			ImageUtils utils = new ImageUtils();
            int[][] org_bbox = new int[][]{{0, 9, 111, 61, 182}, {0, 95, 138, 143, 197}, {0, 185, 98, 235, 166}, {0, 262, 125, 311, 192}, {0, 334, 89, 382, 156}, {0, 429, 98, 474, 163}, {0, 503, 100, 552, 166}, {0, 588, 115, 637, 177}, {0, 694, 92, 743, 164}, {0, 763, 57, 821, 129}, {0, 827, 71, 869, 129}, {0, 860, 89, 915, 155}, {0, 922, 67, 1001, 155}};
            int padw = 0;
            int padh = 0;
            if (im_h > im_w) {
                padw = new BigDecimal(im_h).subtract(new BigDecimal(im_w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (im_h < im_w) {
                padh = new BigDecimal(im_w).subtract(new BigDecimal(im_h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] bbox = YoloImageUtils.formatLabel(org_bbox);
            bbox = YoloImageUtils.resizeBBox(padw, padh, im_w, im_h, bbox);
            System.out.println(JsonUtils.toJson(org_bbox));
            /**
             * cxcywh to xyxy

             */
            for (int i = 0; i < bbox.length; i++) {
                bbox[i][0] = bbox[i][0];
                int cx = bbox[i][1];
                int cy = bbox[i][2];
                int w = bbox[i][3];
                int h = bbox[i][4];
                bbox[i][1] = cx - w / 2;
                bbox[i][2] = cy - h / 2;
                bbox[i][3] = cx + w / 2;
                bbox[i][4] = cy + h / 2;
            }
            YoloImageUtils.resize(data, outputPath2, img_size, img_size, padw, padh, im_w, im_h, bbox);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void showImg() {
        String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
        String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
        String outputPath = "H:\\voc\\sample_set\\";
        YoloDataLoader loader = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, false);
        Tensor imgSet = loader.getImgSet();
        Tensor labelSet = loader.getLabelSet();
        System.out.println("load data finish.");
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < imgSet.number; b++) {
            float[] once = imgSet.getByNumber(b);
            float[] label = labelSet.getByNumber(b);
            int[][] bbox = new int[][]{{0, (int) label[1], (int) label[2], (int) label[3], (int) label[4]}};
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h), im_w, im_h, bbox);
        }
    }

    public int[][] tensorToRGB(Tensor t) {
        int[][] rgb = new int[t.channel][t.height * t.width];
        for (int c = 0; c < t.channel; c++) {
            for (int i = 0; i < rgb.length; i++) {
                rgb[c][i] = (int) t.data[c * t.height * t.width + i];
            }
        }
        return rgb;
    }

    public void yolov1() {
        try {
            String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
            String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
            String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
            String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
            YoloDataLoader trainData = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, false);
            YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, true);
            //			DataSet trainSet = YoloLabelUtils.formatToYolo(trainData.getDataSet(), im_w, im_h);
            DataSet vailSet = YoloLabelUtils.formatToYolo(vailData.getDataSet(), im_w, im_h);
            System.out.println("load data finish.");
            CNN netWork = new CNN(LossType.yolo, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.01f;
            int channel = 3;
            int height = 256;
            int width = 256;
            InputLayer inputLayer = new InputLayer(channel, height, width);
            ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 7, 7, 3, 2, false);
            BNLayer bn1 = new BNLayer();
            LeakyReluLayer active1 = new LeakyReluLayer();
            PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            /**
             * block1  64 * 32 * 32

             */
            BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, 1, netWork);
            LeakyReluLayer active2 = new LeakyReluLayer();
            /**
             * block2  64 * 32 * 32

             */
            BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, 1, netWork);
            LeakyReluLayer active3 = new LeakyReluLayer();
            /**
             * block3  128 * 16 * 16
             * downSample 32 / 2 = 16

             */
            BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, 2, netWork);
            LeakyReluLayer active4 = new LeakyReluLayer();
            /**
             * block4  128 * 16 * 16

             */
            BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, 1, netWork);
            LeakyReluLayer active5 = new LeakyReluLayer();
            /**
             * block5  256 * 8 * 8
             * downSample 16 / 2 = 8

             */
            BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, 2, netWork);
            LeakyReluLayer active6 = new LeakyReluLayer();
            /**
             * block6  256 * 8 * 8

             */
            BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, 1, netWork);
            LeakyReluLayer active7 = new LeakyReluLayer();
            /**
             * block7  512 * 4 * 4
             * downSample 8 / 2 = 4

             */
            BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, 2, netWork);
            LeakyReluLayer active8 = new LeakyReluLayer();
            /**
             * block8  512 * 4 * 4

             */
            BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, 1, netWork);
            LeakyReluLayer active9 = new LeakyReluLayer();
            PoolingLayer pool2 = new PoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight, 3, 3, 2, PoolingType.MAX_POOLING);
            ConvolutionLayer conv2 = new ConvolutionLayer(pool2.oChannel, 1024, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
            BNLayer bn2 = new BNLayer();
            LeakyReluLayer active11 = new LeakyReluLayer();
            ConvolutionLayer conv3 = new ConvolutionLayer(conv2.oChannel, 256, conv2.oWidth, conv2.oHeight, 3, 3, 1, 1, false);
            BNLayer bn3 = new BNLayer();
            LeakyReluLayer active12 = new LeakyReluLayer();
            /**
             * fully  512 * 7 * 7

             */
            int fInputCount = conv3.oChannel * conv3.oWidth * conv3.oHeight;
            FullyLayer full1 = new FullyLayer(fInputCount, 539, true);
            SigmodLayer active13 = new SigmodLayer();
            netWork.addLayer(inputLayer);
            netWork.addLayer(conv1);
            netWork.addLayer(bn1);
            netWork.addLayer(active1);
            netWork.addLayer(pool1);
            /**
             * block1  64

             */
            netWork.addLayer(bl1);
            netWork.addLayer(active2);
            netWork.addLayer(bl2);
            netWork.addLayer(active3);
            /**
             * block2  128

             */
            netWork.addLayer(bl3);
            netWork.addLayer(active4);
            netWork.addLayer(bl4);
            netWork.addLayer(active5);
            /**
             * block3  256

             */
            netWork.addLayer(bl5);
            netWork.addLayer(active6);
            netWork.addLayer(bl6);
            netWork.addLayer(active7);
            /**
             * block4  512

             */
            netWork.addLayer(bl7);
            netWork.addLayer(active8);
            netWork.addLayer(bl8);
            netWork.addLayer(active9);
            netWork.addLayer(pool2);
            netWork.addLayer(conv2);
            netWork.addLayer(bn2);
            netWork.addLayer(active11);
            netWork.addLayer(conv3);
            netWork.addLayer(bn3);
            netWork.addLayer(active12);
            netWork.addLayer(full1);
            netWork.addLayer(active13);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1000, 0.001f, 32, LearnRateUpdate.GD_GECAY, false);
            long start = System.currentTimeMillis();
            optimizer.trainObjectRecognition(trainData.getDataSet(), vailSet, true);
            /**
             * 处理测试预测结果

             */
            float[][][] draw_bbox = optimizer.showObjectRecognition(vailSet, 32);
            YoloDataLoader testData = new YoloDataLoader(trainPath, trainLabelPath, 100, 3, 256, 256, 5, LabelType.csv, false);
            String outputPath = "H:\\voc\\banana-detection\\test_resnet\\";
            showImg(outputPath, testData.getDataSet(), 1, draw_bbox, false);
            System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            try {
                CUDAMemoryManager.freeAll();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public void yolov1_show() {
        int im_w = 256;
        int im_h = 256;
        int classNum = 1;
        int batchSize = 64;
        String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
        String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
        String outputPath = "H:\\voc\\banana-detection\\show\\";
        //		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        //		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
        YoloDataTransform2 dt = new YoloDataTransform2(classNum, DataType.yolov1, 7);
        DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, classNum, batchSize, DataType.yolov1, dt);
        Tensor input = new Tensor(batchSize, 3, im_w, im_h);
        Tensor label = trainData.initLabelTensor();
        for (int i = 0; i < 15; i++) {
            int[] indexs = MatrixUtils.orderInt(batchSize, i);
            trainData.showImg(outputPath, indexs, input, label);
        }
    }

    public void yolov1_tiny() {
        try {
            int im_w = 256;
            int im_h = 256;
            int classNum = 1;
            int batchSize = 64;
            String cfg_path = "H:/voc/train/yolov1-tiny.cfg";
            String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
            String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
            String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
            String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
            YoloDataTransform2 dt = new YoloDataTransform2(classNum, DataType.yolov1, 7);
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, classNum, batchSize, DataType.yolov1, dt);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.csv, im_w, im_h, classNum, batchSize, DataType.yolov1);
            System.out.println("load data finish.");
            Yolo netWork = new Yolo(LossType.yolo, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 3000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            long start = System.currentTimeMillis();
            optimizer.trainObjectRecognition(trainData, vailData);
            //			/**
            //			 * 处理测试预测结果
            //			 */
            //			float[][][] draw_bbox = optimizer.showObjectRecognition(vailSet, batchSize);
            //
            //			YoloDataLoader testData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, false);
            //
            //			String outputPath = "H:\\voc\\banana-detection\\test\\";
            //
            //			showImg(outputPath, testData.getDataSet(), 1, draw_bbox, false);
            System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            try {
                CUDAMemoryManager.freeAll();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public void yolov1_tiny_voc() {
        int im_w = 448;
        int im_h = 448;
        int batchSize = 16;
        int classNum = 20;
        try {
            String cfg_path = "H:/voc/train/yolov1-tiny-voc.cfg";
            String trainPath = "H:\\voc\\train\\imgs";
            String trainLabelPath = "H:\\voc\\train\\labels\\yolov3.txt";
            String testPath = "H:\\voc\\test\\imgs";
            String testLabelPath = "H:\\voc\\test\\labels\\yolov3.txt";
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov1);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov1);
            Yolo netWork = new Yolo(LossType.yolo, UpdaterType.adamw);
            netWork.setClass_num(classNum);
            netWork.CUDNN = true;
            netWork.learnRate = 0.01f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.trainObjectRecognition(trainData, vailData);
            //			/**
            //			 * 处理测试预测结果
            //			 */
            //			float[][][] draw_bbox = optimizer.showObjectRecognition(vailData, batchSize, classNum);
            //
            //			YoloDataLoader testData = new YoloDataLoader(testPath, testLabelPath, batchSize, 3, im_w, im_h, 5, LabelType.text_v3, false);
            //
            //			String outputPath = "H:\\voc\\test\\vail\\";
            //
            //			showImg(outputPath, testData, classNum, draw_bbox, im_w, im_h, false);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testTransforms() {
        String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
        YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, false);
        YoloLabelUtils u = new YoloLabelUtils(1, 4);
        u.transforms(vailData.getImgSet(), vailData.getLabelSet());
        String outputPath = "H:\\voc\\banana-detection\\transforms\\";
        ImageUtils utils = new ImageUtils();
        vailData.setImgSet(vailData.getImgSet());
        for (int b = 0; b < vailData.number; b++) {
            float[] once = vailData.getImgSet().getByNumber(b);
            float[] label = vailData.getLabelSet().getByNumber(b);
            once = MatrixOperation.multiplication(once, 255.0f);
            int[][] bbox = new int[][]{{0, (int) label[1], (int) label[2], (int) label[3], (int) label[4]}};
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h), im_w, im_h, bbox);
        }
    }

    public void testYoloBBox() {
        String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
        YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, true);
        YoloLabelUtils.showLabel(vailData.getLabelSet().data, 100, 7, im_w, im_h);
    }
}

