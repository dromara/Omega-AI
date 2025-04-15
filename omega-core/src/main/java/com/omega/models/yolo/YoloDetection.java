package com.omega.models.yolo;

import java.io.Serializable;

/**
 * 预测对象
 *
 * @author Administrator
 */
public class YoloDetection implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -7047429966674553173L;
    private float[] bbox;
    private float objectness = 0.0f;
    private float classes = 0.0f;
    private float[] prob;
    private int sortClass = 0;

    public YoloDetection(int class_num) {
        this.prob = new float[class_num];
    }

    public float[] getBbox() {
        return bbox;
    }

    public void setBbox(float[] bbox) {
        this.bbox = bbox;
    }

    public float getObjectness() {
        return objectness;
    }

    public void setObjectness(float objectness) {
        this.objectness = objectness;
    }

    public float getClasses() {
        return classes;
    }

    public void setClasses(float classes) {
        this.classes = classes;
    }

    public float[] getProb() {
        return prob;
    }

    public void setProb(float[] prob) {
        this.prob = prob;
    }

    public int getSortClass() {
        return sortClass;
    }

    public void setSortClass(int sortClass) {
        this.sortClass = sortClass;
    }
}

