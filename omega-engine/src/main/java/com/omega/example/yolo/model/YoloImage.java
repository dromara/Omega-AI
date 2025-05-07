package com.omega.example.yolo.model;

import java.io.Serializable;

/**
 * yolo image data
 *
 * @author Administrator
 */
public class YoloImage implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -6521808869204409723L;
    private String name;
    private int channel;
    private int height;
    private int width;
    private int[][] bbox;
    private int[] data;
    private float[] yoloLabel;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getChannel() {
        return channel;
    }

    public void setChannel(int channel) {
        this.channel = channel;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int[] getData() {
        return data;
    }

    public void setData(int[] data) {
        this.data = data;
    }

    public float[] getYoloLabel() {
        return yoloLabel;
    }

    public void setYoloLabel(float[] yoloLabel) {
        this.yoloLabel = yoloLabel;
    }

    public int[][] getBbox() {
        return bbox;
    }

    public void setBbox(int[][] bbox) {
        this.bbox = bbox;
    }
}

