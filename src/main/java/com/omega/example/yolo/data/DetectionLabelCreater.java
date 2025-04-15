package com.omega.example.yolo.data;

import com.omega.common.config.Tensor;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

public class DetectionLabelCreater implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 7572932116255955691L;
    private int box_size = 4;
    private int conf_size = 1;
    private int class_num = 1;
    private int stride = 7;
    private Map<String, List<float[]>> orgLabelData;
    private Tensor label;
    private DataType dataType;
    private int img_w;
    private int img_h;

    public DetectionLabelCreater(Map<String, List<float[]>> orgLabelData, Tensor label, int class_num, int stride, int img_w, int img_h, DataType dataType) {
        this.orgLabelData = orgLabelData;
        this.label = label;
        this.class_num = class_num;
        this.stride = stride;
        this.dataType = dataType;
    }

    public void clearData() {
        this.label.data = new float[this.label.dataLength];
    }

    public void loadLabel(String key, int index) {
        switch (dataType) {
            case yolov1:
                loadLabelToYolov1(key, index);
                break;
            case yolov3:
                loadLabelToYolov3(key, index);
                break;
        }
    }

    private void loadLabelToYolov1(String key, int index) {
        int once = (box_size + conf_size + class_num);
        int oneSize = stride * stride * once;
        List<float[]> list = orgLabelData.get(key);
        for (int n = 0; n < list.size(); n++) {
            float[] box = list.get(n);
            int clazz = (int) box[0] + 1;
            float cx = box[1] / img_w;
            float cy = box[2] / img_h;
            float w = box[3] / img_w;
            float h = box[4] / img_h;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            label.data[index * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
            /**
             * class

             */
            label.data[index * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            label.data[index * oneSize + gridy * stride * once + gridx * once + class_num + 1] = px;
            label.data[index * oneSize + gridy * stride * once + gridx * once + class_num + 2] = py;
            label.data[index * oneSize + gridy * stride * once + gridx * once + class_num + 3] = w;
            label.data[index * oneSize + gridy * stride * once + gridx * once + class_num + 4] = h;
        }
    }

    private void loadLabelToYolov3(String key, int index) {
    }

    public int getBox_size() {
        return box_size;
    }

    public void setBox_size(int box_size) {
        this.box_size = box_size;
    }

    public int getConf_size() {
        return conf_size;
    }

    public void setConf_size(int conf_size) {
        this.conf_size = conf_size;
    }

    public int getClass_num() {
        return class_num;
    }

    public void setClass_num(int class_num) {
        this.class_num = class_num;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public Map<String, List<float[]>> getOrgLabelData() {
        return orgLabelData;
    }

    public void setOrgLabelData(Map<String, List<float[]>> orgLabelData) {
        this.orgLabelData = orgLabelData;
    }

    public Tensor getLabel() {
        return label;
    }

    public void setLabel(Tensor label) {
        this.label = label;
    }

    public DataType getDataType() {
        return dataType;
    }
}

