package com.omega.example.yolo.utils;

import com.omega.common.utils.MatrixUtils;

import java.io.Serializable;

/**
 * omega engine image class
 *
 * @author Administrator
 */
public class OMImage implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -5416036227388123219L;
    private int channel = 0;
    private int height = 0;
    private int width = 0;
    private float[] data;

    public OMImage(int channel, int height, int width, float[] data) {
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.data = data;
    }

    public void clear() {
        MatrixUtils.zero(data);
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

    public float[] getData() {
        return data;
    }

    public void setData(float[] data) {
        this.data = data;
    }
}

