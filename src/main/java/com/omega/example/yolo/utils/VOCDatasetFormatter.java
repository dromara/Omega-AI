package com.omega.example.yolo.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import com.omega.example.yolo.data.DataType;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.data.ImageLoader;

/**
 * voc数据集处理工具
 * 1.读取Annotations的xml，生成bbox文件.
 * 2.读取原始图片并转换图片大小h:416,w:416.同时生成转换后的bbox坐标文件：rlabels.txt
 * 3.读取train.txt和vail.txt生成训练标签和测试标签集(train_label.txt,vail_label.txt)
 */
public class VOCDatasetFormatter {
	
	public static void main(String[] args) {
		
		/**
		 * 1.读取Annotations的xml，生成bbox文件.
		 */
		create_bbox_by_xml();
		
		/**
		 * 2.读取原始图片并转换图片大小h:416,w:416.同时生成转换后的bbox坐标文件：rlabels.txt
		 */
		imageFormat();
		
		/**
		 * 3.读取train.txt和vail.txt生成训练标签和测试标签集(train_label.txt,vail_label.txt)
		 */
		createTrainVailLabel();
		
	}
	
	public static void create_bbox_by_xml(){
		String rootPath = "D:\\dataset\\VOCData\\";
		String imgDir = rootPath + "\\JPEGImages";
		String labelDir = rootPath + "\\Annotations";
		String labelPath = rootPath + "\\labels.txt";
		String bboxPath = rootPath + "\\bbox.txt";
		YoloImageUtils.xml2Yolo(imgDir, labelDir, labelPath, bboxPath);
	}
	
	public static void imageFormat() {
		try {
            String imgDirPath = "D:\\dataset\\VOCData\\JPEGImages";
            String labelPath = "D:\\dataset\\VOCData\\bbox.txt";
            String outputDirPath = "D:\\dataset\\VOCData\\resized\\imgs\\";
            String labelTXTPath = "D:\\dataset\\VOCData\\resized\\rlabels.txt";
            int width = 416;
            int height = 416;
            Map<String, float[]> labelMap = ImageLoader.loadLabelDataForTXT(labelPath);
            Map<String, float[]> rlabelMap = new HashMap<String, float[]>();
            String[] names = new String[labelMap.size()];
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                int i = 0;
                for (File img : file.listFiles()) {
                    String key = img.getName().split("\\.")[0];
                    float[] rlabel = ImageLoader.resizeImage(img, labelMap.get(key), width, height, outputDirPath + img.getName());
                    rlabelMap.put(key, rlabel);
                    names[i] = key;
                    i++;
                }
            }
            File txt = new File(labelTXTPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);) {
                for (String name : names) {
                    String text = name;
                    for (float val : (float[]) rlabelMap.get(name)) {
                        text += " " + Math.round(val);
                    }
                    text += "\n";
                    //					System.out.println(text);
                    fos.write(text.getBytes());
                }
                fos.flush();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
	}
	
	public static void createTrainVailLabel() {
		try {
            int im_w = 416;
            int im_h = 416;
            int classNum = 5;
            int batchSize = 64;
            String trainDataPath = "D:\\dataset\\VOCData\\train.txt";
            String testDataPath = "D:\\dataset\\VOCData\\test.txt";
            String orgPath = "D:\\dataset\\VOCData\\resized\\imgs\\";
            String orgLabelPath = "D:\\dataset\\VOCData\\resized\\rlabels.txt";
            String trainPath = "D:\\dataset\\VOCData\\resized\\train\\";
            String vailPath = "D:\\dataset\\VOCData\\resized\\vail\\";
            String trainLabelPath = "D:\\dataset\\VOCData\\resized\\train_label.txt";
            String vailLabelPath = "D:\\dataset\\VOCData\\resized\\vail_label.txt";
            DetectionDataLoader orgData = new DetectionDataLoader(orgPath, orgLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3);
            int trainSize = 16417;
            int testSize = 2052;
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

             */
            createLabelTXT(trainLabelPath, trainLabelData);
            createLabelTXT(vailLabelPath, testLabelData);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
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
	
}
