package com.omega.example.yolo.data;

import com.omega.common.utils.MathUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.LabelFileType;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DetectionDataLoader
 *
 * @author Administrator
 */
public class DetectionDataLoader extends BaseDataLoader {
    public static float[] mean = new float[]{0.491f, 0.482f, 0.446f};
    public static float[] std = new float[]{0.247f, 0.243f, 0.261f};
    public Map<String, float[]> orgLabelData;
    public String[] idxSet;
    private String imgDirPath;
    private String labelPath;
    private boolean dataEnhance = false;
    private LabelFileType labelFileType;
    private int onceLabelSize = 5;
    private int img_w;
    private int img_h;
    private DataType dataType;
    private int classNum = 1;
    private int stride = 7;
    private String extName;
    private int bboxNum = 90;
    private DataTransform dataTransform;

    public DetectionDataLoader(String imgDirPath, String labelPath, LabelFileType labelFileType, int img_w, int img_h, int classNum, int batchSize, DataType dataType) {
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.labelFileType = labelFileType;
        this.img_w = img_w;
        this.img_h = img_h;
        this.dataType = dataType;
        this.classNum = classNum;
        this.batchSize = batchSize;
        init();
    }

    public DetectionDataLoader(String imgDirPath, String labelPath, LabelFileType labelFileType, int img_w, int img_h, int classNum, int batchSize, DataType dataType, int onceLabelSize) {
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.labelFileType = labelFileType;
        this.onceLabelSize = onceLabelSize;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.dataType = dataType;
        this.classNum = classNum;
        init();
    }

    public DetectionDataLoader(String imgDirPath, String labelPath, LabelFileType labelFileType, int img_w, int img_h, int classNum, int batchSize, DataType dataType, DataTransform dataTransform) {
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.labelFileType = labelFileType;
        if (dataTransform != null) {
            this.dataEnhance = true;
        }
        this.dataTransform = dataTransform;
        this.img_w = img_w;
        this.img_h = img_h;
        this.dataType = dataType;
        this.classNum = classNum;
        this.batchSize = batchSize;
        init();
    }

    public DetectionDataLoader(String imgDirPath, String labelPath, LabelFileType labelFileType, int img_w, int img_h, int classNum, int batchSize, DataType dataType, int onceLabelSize, DataTransform dataTransform) {
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.labelFileType = labelFileType;
        this.onceLabelSize = onceLabelSize;
        if (dataTransform != null) {
            this.dataEnhance = true;
        }
        this.dataTransform = dataTransform;
        this.img_w = img_w;
        this.img_h = img_h;
        this.dataType = dataType;
        this.classNum = classNum;
        this.batchSize = batchSize;
        init();
    }

    public void init() {
        loadFileCount();
        if (this.labelPath != null) {
            loadLabelData();
        } else {
            loadFileName();
        }
    }

    public void loadFileName() {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                String[] filenames = file.list();
                for (int i = 0; i < idxSet.length; i++) {
                    this.idxSet[i] = filenames[i].split("\\.")[0];
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadFileCount() {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                String[] filenames = file.list();
                this.number = filenames.length;
                //				System.out.println(number);
                this.orgLabelData = new HashMap<String, float[]>(number);
                this.idxSet = new String[number];
                this.extName = filenames[0].split("\\.")[1];
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadLabelData() {
        switch (labelFileType) {
            case txt:
                loadLabelDataForTXT();
                break;
            case csv:
                loadLabelDataForCSV();
                break;
            default:
                break;
        }
    }

    public void showImg(String outputPath, int[] indexs, Tensor input, Tensor label) {
        if (dataType == DataType.yolov3) {
            //			System.out.println(label.getDataLength());
            ImageLoader.load(imgDirPath, extName, input, label, idxSet, indexs, orgLabelData, bboxNum, classNum);
            ImageLoader.showImg(input, label, classNum, outputPath);
        } else {
            FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, dataEnhance);
            if (dataTransform != null) {
                dataTransform.showTransform(outputPath, input, label, idxSet, indexs, orgLabelData);
            }
        }
    }

    public void loadLabelDataForCSV() {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    List<float[]> once = new ArrayList<float[]>();
                    this.idxSet[idx - 1] = list[0];
                    int page = (list.length - 1) / this.onceLabelSize;
                    for (int i = 0; i < page; i++) {
                        float[] bbox = new float[this.onceLabelSize];
                        for (int j = 0; j < this.onceLabelSize; j++) {
                            bbox[j] = Float.parseFloat(list[i * this.onceLabelSize + j + 1]);
                        }
                        once.add(bbox);
                    }
                    /**
                     * 组装bbox

                     */
                    float[] r = new float[once.size() * 5];
                    for (int i = 0; i < once.size(); i++) {
                        for (int j = 0; j < once.get(i).length; j++) {
                            r[i * 5 + j] = once.get(i)[j];
                        }
                    }
                    this.orgLabelData.put(list[0], r);
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadLabelDataForTXT() {
        try (FileInputStream fin = new FileInputStream(this.labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                String[] list = strTmp.split(" ");
                List<float[]> once = new ArrayList<float[]>();
                int page = (list.length - 1) / this.onceLabelSize;
                for (int i = 0; i < page; i++) {
                    float[] bbox = new float[this.onceLabelSize];
                    for (int j = 0; j < this.onceLabelSize; j++) {
                        bbox[j] = Float.parseFloat(list[i * this.onceLabelSize + j + 1]);
                    }
                    once.add(bbox);
                }
                /**
                 * 组装bbox

                 */
                float[] r = new float[once.size() * 5];
                for (int i = 0; i < once.size(); i++) {
                    for (int j = 0; j < once.get(i).length; j++) {
                        r[i * 5 + j] = once.get(i)[j];
                    }
                }
                this.orgLabelData.put(list[0], r);
                this.idxSet[idx] = list[0];
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        if (dataType == DataType.yolov3) {
            ImageLoader.loadVail(imgDirPath, extName, input, label, idxSet, indexs, orgLabelData, bboxNum, classNum);
        } else {
            /**
             * 根据数据类型加载标签数据

             */
            loadLabelToDataType(indexs, label, dataType);
            /**
             * 加载input数据

             */
            FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        if (dataType == DataType.yolov3) {
            ImageLoader.loadVail(imgDirPath, extName, input, null, idxSet, indexs, orgLabelData, bboxNum, classNum);
        } else {
            /**
             * 加载input数据

             */
            FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        String filePath = imgDirPath + "/" + idxSet[index];
        if (!filePath.contains(".")) {
            filePath += ".jpg";
        }
        if (dataType == DataType.yolov3) {
            return ImageLoader.resized(filePath, this.img_w, this.img_h);
        } else {
            return YoloImageUtils.loadImgDataToArray(filePath);
        }
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        //		long start2 = System.nanoTime();
        if (dataType == DataType.yolov3) {
            ImageLoader.load(imgDirPath, extName, input, label, idxSet, indexs, orgLabelData, bboxNum, classNum);
        } else {
            /**
             * 加载input数据

             */
            FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, dataEnhance);
            if (dataTransform != null) {
                /**
                 * 数据增强

                 */
                dataTransform.transform(input, label, idxSet, indexs, orgLabelData);
            } else {
                /**
                 * 根据数据类型加载标签数据

                 */
                loadLabelToDataType(indexs, label, dataType);
            }
        }
        //		System.out.println((System.nanoTime() - start2)/1e6+"ms.");
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void normalization(Tensor input) {
        for (int i = 0; i < input.getDataLength(); i++) {
            int f = (i / input.getShape()[3] / input.getShape()[2]) % input.getShape()[1];
            input.getData()[i] = (input.getData()[i] - mean[f]) / std[f];
        }
    }

    public void fileDataLoader(int[] indexs, Tensor input) {
        for (int i = 0; i < batchSize; i++) {
            String filePath = imgDirPath + "/" + idxSet[indexs[i]] + "." + extName;
            YoloImageUtils.loadImgDataToTensor(filePath, input, i);
            //			System.out.println(filePath);
        }
    }

    public void loadLabelToDataType(int[] indexs, Tensor label, DataType dataType) {
        switch (dataType) {
            case yolov1:
                loadLabelToYolov1(indexs, label);
                break;
            case yolov3:
                loadLabelToYolov3(indexs, label, 90);
                break;
        }
    }

    public void loadLabelToYolov1(int[] indexs, Tensor label) {
        int once = (5 + classNum);
        int oneSize = stride * stride * once;
        label.setData(new float[label.getShape()[0] * oneSize]);
        for (int b = 0; b < label.getShape()[0]; b++) {
            String key = idxSet[indexs[b]];
            float[] box = orgLabelData.get(key);
            for (int n = 0; n < box.length / 5; n++) {
                int clazz = (int) box[n * 5 + 0] + 1;
                float x1 = box[n * 5 + 1] / img_w;
                float y1 = box[n * 5 + 2] / img_h;
                float x2 = box[n * 5 + 3] / img_w;
                float y2 = box[n * 5 + 4] / img_h;
                float cx = (x1 + x2) / 2;
                float cy = (y1 + y2) / 2;
                float w = (x2 - x1);
                float h = (y2 - y1);
                int gridx = (int) (cx * stride);
                int gridy = (int) (cy * stride);
                float px = cx * stride - gridx;
                float py = cy * stride - gridy;
                /**
                 * c1
                 */
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
                /**
                 * class
                 */
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
                /**
                 * x1,y1,w1,h1
                 */
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + classNum + 1] = px;
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + classNum + 2] = py;
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + classNum + 3] = w;
                label.getData()[b * oneSize + gridy * stride * once + gridx * once + classNum + 4] = h;
            }
        }
    }

    public void loadLabelToYolov3(int[] indexs, Tensor label, int bbox_num) {
        label.setData(new float[label.getData().length]);
        for (int i = 0; i < label.getShape()[0]; i++) {
            String key = idxSet[indexs[i]];
            float[] box = orgLabelData.get(key);
            for (int c = 0; c < box.length / 5; c++) {
                float clazz = box[c * 5 + 0];
                float x1 = box[c * 5 + 1] / img_w;
                float y1 = box[c * 5 + 2] / img_h;
                float x2 = box[c * 5 + 3] / img_w;
                float y2 = box[c * 5 + 4] / img_h;
                float cx = (x1 + x2) / 2;
                float cy = (y1 + y2) / 2;
                float w = (x2 - x1);
                float h = (y2 - y1);
                label.getData()[i * bbox_num * 5 + c * 5 + 0] = cx;
                label.getData()[i * bbox_num * 5 + c * 5 + 1] = cy;
                label.getData()[i * bbox_num * 5 + c * 5 + 2] = w;
                label.getData()[i * bbox_num * 5 + c * 5 + 3] = h;
                label.getData()[i * bbox_num * 5 + c * 5 + 4] = clazz;
            }
        }
    }

    public Tensor initLabelTensor() {
        switch (dataType) {
            case yolov1:
                return new Tensor(batchSize, 1, 1, (5 + classNum) * stride * stride, true);
            case yolov3:
                return new Tensor(batchSize, 1, 1, 5 * 90, true);
        }
        return null;
    }

    public int[] getIndexsByAsc(int pageIndex, int batchSize) {
        int start = pageIndex * batchSize;
        int end = pageIndex * batchSize + batchSize;
        if (end > number) {
            start = start - (end - number);
        }
        int[] indexs = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indexs[i] = start + i;
        }
        return indexs;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
    }
}

