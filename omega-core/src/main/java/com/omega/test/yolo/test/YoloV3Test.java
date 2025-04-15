package com.omega.test.yolo.test;

import com.omega.common.tensor.Tensor;
import com.omega.utils.image.ImageUtils;
import com.omega.utils.JsonUtils;
import com.omega.utils.MatrixOperation;
import com.omega.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.DarknetLoader;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.data.yolo.DataType;
import com.omega.data.yolo.DetectionDataLoader;
import com.omega.data.yolo.YoloDataTransform2;
import com.omega.models.yolo.YoloBox;
import com.omega.models.yolo.YoloDetection;
import com.omega.utils.yolo.AnchorBoxUtils;
import com.omega.utils.yolo.LabelFileType;
import com.omega.utils.yolo.LabelUtils;

import java.io.*;
import java.util.*;

public class YoloV3Test {
    public static void showImg(String outputPath, DetectionDataLoader dataSet, int class_num, List<YoloBox> score_bbox, int batchSize, boolean format, int im_w, int im_h, String[] labelset) {
        ImageUtils utils = new ImageUtils();
        int lastIndex = dataSet.number % batchSize;
        for (int b = 0; b < dataSet.number; b++) {
            float[] once = dataSet.loadData(b);
            once = MatrixOperation.multiplication(once, 255.0f);
            int bbox_index = b;
            if (b >= (dataSet.number - lastIndex)) {
                bbox_index = b + (batchSize - lastIndex);
            }
            YoloBox box = score_bbox.get(bbox_index);
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < box.getDets().size(); l++) {
                if (box.getDets().get(l) != null && box.getDets().get(l).getObjectness() > 0 && !MatrixUtils.isZero(box.getDets().get(l).getProb())) {
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                YoloDetection det = box.getDets().get(index);
                bbox[i][0] = (int) det.getClasses();
                bbox[i][1] = (int) ((det.getBbox()[0] - det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][2] = (int) ((det.getBbox()[1] - det.getBbox()[3] / 2.0f) * im_h);
                bbox[i][3] = (int) ((det.getBbox()[0] + det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][4] = (int) ((det.getBbox()[1] + det.getBbox()[3] / 2.0f) * im_h);
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox, labelset);
        }
    }

    public static void copyFileUsingStream(File source, File dest) throws IOException {
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new FileInputStream(source);
            os = new FileOutputStream(dest);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            is.close();
            os.close();
        }
    }

    public static void createLabelTXT(String txtPath, Map<String, float[]> data) {
        File txt = new File(txtPath);
        if (!txt.exists()) {
            try {
                txt.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            } // 创建新文件,有同名的文件的话直接覆盖
        }
        try (FileOutputStream fos = new FileOutputStream(txt);) {
            for (String name : data.keySet()) {
                String text = name;
                for (float val : data.get(name)) {
                    text += " " + Math.round(val);
                }
                text += "\n";
                fos.write(text.getBytes());
            }
            fos.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            YoloV3Test y = new YoloV3Test();
            //			y.yolov3_tiny_voc();
            //			y.yolov3_tiny();
            //			y.getAnchors();
            //			y.yolov3_show();
            //			y.createTrainTestDataSet();
            //			y.yolov3_show();
            //			y.yolov3_show2();
            //			y.createMaskTrainTestDataSet();
            y.yolov3_tiny_mask();
            //			y.yolov3_tiny_helmet();
            //			y.yolov3_tiny_yz();
            //			y.yolov3_tiny_voc();
            //			y.yolov3_tiny_sm();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void yolov3_tiny() {
        int im_w = 256;
        int im_h = 256;
        int batchSize = 64;
        int class_num = 1;
        try {
            String cfg_path = "H:\\voc\\banana-detection\\yolov3-tiny-banana.cfg";
            String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
            String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
            String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
            String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
            YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{200, 500, 1000, 1200, 2000};
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理测试预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\banana-detection\\test_yolov3\\";
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, null);
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

    public void yolov3_tiny_helmet() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 5;
        String[] labelset = new String[]{"none", "white", "yellow", "blue", "red"};
        try {
            String cfg_path = "H:\\voc\\helmet_dataset\\yolov3-tiny-helmet.cfg";
            String trainPath = "H:\\voc\\helmet\\resized\\train";
            String trainLabelPath = "H:\\voc\\helmet\\resized\\train_label.txt";
            String testPath = "H:\\voc\\helmet\\resized\\vail";
            String testLabelPath = "H:\\voc\\helmet\\resized\\vail_label.txt";
            String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.conv.15";
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            DarknetLoader.loadWeight(netWork, weightPath, 14, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理测试预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\helmet\\test_yolov3\\";
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
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

    public void yolov3_tiny_sm() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 113;
        String[] labelset = new String[113];
        try {
            String cfg_path = "H:\\voc\\sm\\resized\\yolov3-tiny-sm.cfg";
            String labelPath = "H:\\voc\\\\sm\\VOC\\labels.txt";
            String trainPath = "H:\\voc\\sm\\resized\\train";
            String trainLabelPath = "H:\\voc\\sm\\resized\\train_label.txt";
            String testPath = "H:\\voc\\sm\\resized\\vail";
            String testLabelPath = "H:\\voc\\sm\\resized\\vail_label.txt";
            String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.conv.15";
            try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
                String strTmp = "";
                int idx = 0;
                while ((strTmp = buffReader.readLine()) != null) {
                    labelset[idx] = strTmp;
                    idx++;
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            DarknetLoader.loadWeight(netWork, weightPath, 14, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理测试预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\sm\\test_yolov3\\";
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
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

    public void yolov3_tiny_voc() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 20;
        String[] labelset = new String[]{"person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"};
        try {
            String cfg_path = "H:/voc/train/yolov3-tiny-voc.cfg";
            String trainPath = "H:\\voc\\train\\resized\\imgs";
            String trainLabelPath = "H:\\voc\\train\\resized\\rlabels.txt";
            String testPath = "H:\\voc\\test\\resized\\imgs";
            String testLabelPath = "H:\\voc\\test\\resized\\rlabels.txt";
            String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.conv.15";
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            DarknetLoader.loadWeight(netWork, weightPath, 14, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 3000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            //			optimizer.lr_step = new int[] {200,500,1000,1500,2000};
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理测试预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\test\\resized\\test_yolov3\\";
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
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

    public void getAnchors() {
        try {
            String trainLabelPath = "H:\\voc\\mask\\data\\dataset\\labels\\train_label.txt";
            Tensor bbox = LabelUtils.loadBoxTXT(trainLabelPath);
            Tensor anchors = AnchorBoxUtils.getAnchorBox(bbox, 6);
            for (int i = 0; i < 6; i++) {
                float w = anchors.data[i * 2 + 0];
                float h = anchors.data[i * 2 + 1];
                System.out.println("w:" + w + "*h:" + h + "=" + w * h);
            }
            System.out.println(JsonUtils.toJson(anchors.data));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void yolov3_show() {
        int im_w = 416;
        int im_h = 416;
        int classNum = 1;
        int batchSize = 167;
        //		String trainPath = "H:\\voc\\mask\\data\\JPEGImages";
        //		String trainLabelPath = "H:\\voc\\mask\\data\\labels.txt";
        String trainPath = "H:\\voc\\yz\\seal\\resized\\train";
        String trainLabelPath = "H:\\voc\\yz\\seal\\resized\\train_label.txt";
        //		String outputPath = "H:\\voc\\mask\\data\\show\\";
        String outputPath = "H:\\voc\\yz\\seal\\resized\\show\\";
        //		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        //		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
        DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3);
        Tensor input = new Tensor(batchSize, 3, im_w, im_h);
        Tensor label = trainData.initLabelTensor();
        for (int i = 0; i < 1; i++) {
            int[] indexs = MatrixUtils.orderInt(batchSize, i);
            trainData.showImg(outputPath, indexs, input, label);
        }
    }

    public void yolov3_show2() {
        int im_w = 256;
        int im_h = 256;
        int classNum = 1;
        int batchSize = 64;
        String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
        String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
        String outputPath = "H:\\voc\\banana-detection\\bananas_train\\show\\";
        //		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        //		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
        YoloDataTransform2 dt = new YoloDataTransform2(classNum, DataType.yolov3, 90);
        DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, classNum, batchSize, DataType.yolov3, dt);
        Tensor input = new Tensor(batchSize, 3, im_w, im_h);
        Tensor label = trainData.initLabelTensor();
        for (int i = 0; i < 1; i++) {
            int[] indexs = MatrixUtils.orderInt(batchSize, i);
            trainData.showImg(outputPath, indexs, input, label);
        }
    }

    public void yolov3_tiny_mask() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 2;
        String[] labelset = new String[]{"unmask", "mask"};
        try {
            String cfg_path = "H:\\voc\\mask\\data\\\\dataset\\yolov3-tiny-mask.cfg";
            //			String trainPath = "H:\\voc\\mask\\data\\dataset\\train";
            //			String trainLabelPath = "H:\\voc\\mask\\data\\dataset\\labels\\train_label.txt";
            //
            //			String testPath = "H:\\voc\\mask\\data\\dataset\\vail";
            //			String testLabelPath = "H:\\voc\\mask\\data\\dataset\\labels\\vail_label.txt";
            String trainPath = "H:\\voc\\mask\\data\\resized\\train";
            String trainLabelPath = "H:\\voc\\mask\\data\\resized\\train_label.txt";
            String testPath = "H:\\voc\\mask\\data\\resized\\vail";
            String testLabelPath = "H:\\voc\\mask\\data\\resized\\vail_label.txt";
            //			String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.conv.15";
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            //			DarknetLoader.loadWeight(netWork, weightPath, 14, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 300, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{50, 100, 250};
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理测试预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\mask\\data\\resized\\test_yolov3\\";
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
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

    public void yolov3_tiny_yz() {
        int im_w = 416;
        int im_h = 416;
        int batchSize = 24;
        int class_num = 1;
        String[] labelset = new String[1];
        try {
            String cfg_path = "H:\\voc\\yz\\seal\\resized\\yolov3-tiny-yz.cfg";
            String labelPath = "H:\\voc\\yz\\seal\\labels.txt";
            String trainPath = "H:\\voc\\yz\\seal\\resized\\train";
            String trainLabelPath = "H:\\voc\\yz\\seal\\resized\\train_label.txt";
            String vailPath = "H:\\voc\\yz\\seal\\resized\\vail";
            String vailLabelPath = "H:\\voc\\yz\\seal\\resized\\vail_label.txt";
            //			String weightPath = "H:\\voc\\darknet_yolov7\\yolov7-tiny.conv.87";
            try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
                String strTmp = "";
                int idx = 0;
                while ((strTmp = buffReader.readLine()) != null) {
                    labelset[idx] = strTmp;
                    idx++;
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            DetectionDataLoader vailData = new DetectionDataLoader(vailPath, vailLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            ModelLoader.loadConfigToModel(netWork, cfg_path);
            //			DarknetLoader.loadWeight(netWork, weightPath, 86, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{500, 1000, 1500};
            optimizer.trainObjectRecognitionOutputs(trainData, vailData);
            /**
             * 处理验证集预测结果
             *
             */
            List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
            String outputPath = "H:\\voc\\yz\\seal\\resized\\test_yolov3\\";
            //			System.out.println(JsonUtils.toJson(draw_bbox.get(0)));
            showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, labelset);
            System.out.println("vail finish.");
            /**
             * 处理测试集预测结果
             *
             */
            String testPath = "H:\\voc\\yz\\seal\\resized\\test";
            DetectionDataLoader testData = new DetectionDataLoader(testPath, null, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);
            List<YoloBox> test_draw_bbox = optimizer.showObjectRecognitionYoloV3(testData, batchSize);
            //			System.out.println(JsonUtils.toJson(test_draw_bbox.get(0)));
            String testOutputPath = "H:\\voc\\yz\\seal\\resized\\test_result\\";
            showImg(testOutputPath, testData, class_num, test_draw_bbox, batchSize, false, im_w, im_h, labelset);
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

    public void createMaskTrainTestDataSet() {
        try {
            int im_w = 416;
            int im_h = 416;
            int classNum = 5;
            int batchSize = 64;
            String trainDataPath = "H:\\voc\\mask\\data\\train.txt";
            String testDataPath = "H:\\voc\\mask\\data\\test.txt";
            String orgPath = "H:\\voc\\mask\\data\\resized\\imgs\\";
            String orgLabelPath = "H:\\voc\\mask\\data\\resized\\rlabels.txt";
            String trainPath = "H:\\voc\\mask\\data\\resized\\train\\";
            String vailPath = "H:\\voc\\mask\\data\\resized\\vail\\";
            String trainLabelPath = "H:\\voc\\mask\\data\\resized\\train_label.txt";
            String vailLabelPath = "H:\\voc\\mask\\data\\resized\\vail_label.txt";
            DetectionDataLoader orgData = new DetectionDataLoader(orgPath, orgLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3);
            int trainSize = 360;
            int testSize = 38;
            Map<String, float[]> trainLabelData = new HashMap<String, float[]>();
            Map<String, float[]> testLabelData = new HashMap<String, float[]>();
            String[] trainNames = new String[trainSize];
            String[] testNames = new String[testSize];
            try (FileInputStream fin = new FileInputStream(trainDataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
                String strTmp = "";
                int idx = 0;
                while ((strTmp = buffReader.readLine()) != null) {
                    trainNames[idx] = strTmp.split(".jpg")[0];
                    idx++;
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            try (FileInputStream fin = new FileInputStream(testDataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
                String strTmp = "";
                int idx = 0;
                while ((strTmp = buffReader.readLine()) != null) {
                    testNames[idx] = strTmp.split(".jpg")[0];
                    idx++;
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            /**
             * 复制文件
             *
             */
            for (int b = 0; b < trainSize; b++) {
                String filename = trainNames[b];
                System.out.println(filename);
                if (orgData.orgLabelData.get(filename).length <= 450) {
                    File file = new File(orgPath + filename + ".jpg");
                    File outFile = new File(trainPath + filename + ".jpg");
                    copyFileUsingStream(file, outFile);
                    trainLabelData.put(filename, orgData.orgLabelData.get(filename));
                }
            }
            for (int b = 0; b < testSize; b++) {
                String filename = testNames[b];
                if (orgData.orgLabelData.get(filename).length <= 450) {
                    File file = new File(orgPath + filename + ".jpg");
                    File outFile = new File(vailPath + filename + ".jpg");
                    copyFileUsingStream(file, outFile);
                    testLabelData.put(filename, orgData.orgLabelData.get(filename));
                }
            }
            /**
             * 复制label
             *
             */
            createLabelTXT(trainLabelPath, trainLabelData);
            createLabelTXT(vailLabelPath, testLabelData);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void createTrainTestDataSet() {
        try {
            int im_w = 416;
            int im_h = 416;
            int classNum = 5;
            int batchSize = 64;
            float trainRatio = 0.8f;
            String orgPath = "H:\\voc\\sm\\resized\\imgs\\";
            String orgLabelPath = "H:\\voc\\sm\\resized\\rlabels.txt";
            String trainPath = "H:\\voc\\sm\\resized\\train\\";
            String vailPath = "H:\\voc\\sm\\resized\\vail\\";
            String trainLabelPath = "H:\\voc\\sm\\resized\\train_label.txt";
            String vailLabelPath = "H:\\voc\\sm\\resized\\vail_label.txt";
            DetectionDataLoader orgData = new DetectionDataLoader(orgPath, orgLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3);
            List<Integer> list = new ArrayList<Integer>();
            for (int i = 0; i < orgData.idxSet.length; i++) {
                list.add(i);
            }
            Collections.shuffle(list);
            //			System.out.println(JsonUtils.toJson(list));
            int trainSize = Math.round(orgData.idxSet.length * trainRatio);
            System.out.println(trainSize);
            int testSize = orgData.idxSet.length - trainSize;
            Map<String, float[]> trainLabelData = new HashMap<String, float[]>();
            Map<String, float[]> testLabelData = new HashMap<String, float[]>();
            /**
             * 复制文件
             *
             */
            for (int b = 0; b < trainSize; b++) {
                int index = list.get(b);
                String filename = orgData.idxSet[index];
                System.out.println(filename);
                if (orgData.orgLabelData.get(filename).length <= 450) {
                    File file = new File(orgPath + filename + ".jpg");
                    File outFile = new File(trainPath + filename + ".jpg");
                    copyFileUsingStream(file, outFile);
                    trainLabelData.put(filename, orgData.orgLabelData.get(filename));
                }
            }
            for (int b = 0; b < testSize; b++) {
                int index = list.get(b + trainSize);
                String filename = orgData.idxSet[index];
                if (orgData.orgLabelData.get(filename).length <= 450) {
                    File file = new File(orgPath + filename + ".jpg");
                    File outFile = new File(vailPath + filename + ".jpg");
                    copyFileUsingStream(file, outFile);
                    testLabelData.put(filename, orgData.orgLabelData.get(filename));
                }
            }
            /**
             * 复制label
             *
             */
            createLabelTXT(trainLabelPath, trainLabelData);
            createLabelTXT(vailLabelPath, testLabelData);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}
