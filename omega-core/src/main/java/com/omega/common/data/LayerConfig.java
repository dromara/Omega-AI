package com.omega.common.data;

import java.io.Serializable;

import com.omega.engine.tensor.Tensor;

public class LayerConfig implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 402518562731570825L;
    private int index;
    private String layerType;
    private Tensor weight;
    private Tensor bias;
    private float[] runingMean;
    private float[] runingVar;
    private float[] gama;
    private float[] beta;
    private int number;
    private int channel;
    private int height;
    private int width;
    private int oChannel;
    private int oHeight;
    private int oWidth;
    private int padding;
    private int stride;
    private String poolingType;
    private int pWidth = 0;
    private int pHeight = 0;
    private int kNumber = 0;
    private boolean hasBias = true;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public String getLayerType() {
        return layerType;
    }

    public void setLayerType(String layerType) {
        this.layerType = layerType;
    }

    public Tensor getWeight() {
        return weight;
    }

    public void setWeight(Tensor weight) {
        this.weight = weight;
    }

    public Tensor getBias() {
        return bias;
    }

    public void setBias(Tensor bias) {
        this.bias = bias;
    }

    public float[] getRuningMean() {
        return runingMean;
    }

    public void setRuningMean(float[] runingMean) {
        this.runingMean = runingMean;
    }

    public float[] getRuningVar() {
        return runingVar;
    }

    public void setRuningVar(float[] runingVar) {
        this.runingVar = runingVar;
    }

    public float[] getGama() {
        return gama;
    }

    public void setGama(float[] gama) {
        this.gama = gama;
    }

    public float[] getBeta() {
        return beta;
    }

    public void setBeta(float[] beta) {
        this.beta = beta;
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
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

    public int getoChannel() {
        return oChannel;
    }

    public void setoChannel(int oChannel) {
        this.oChannel = oChannel;
    }

    public int getoHeight() {
        return oHeight;
    }

    public void setoHeight(int oHeight) {
        this.oHeight = oHeight;
    }

    public int getoWidth() {
        return oWidth;
    }

    public void setoWidth(int oWidth) {
        this.oWidth = oWidth;
    }

    public boolean isHasBias() {
        return hasBias;
    }

    public void setHasBias(boolean hasBias) {
        this.hasBias = hasBias;
    }

    public int getPadding() {
        return padding;
    }

    public void setPadding(int padding) {
        this.padding = padding;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public String getPoolingType() {
        return poolingType;
    }

    public void setPoolingType(String poolingType) {
        this.poolingType = poolingType;
    }

    public int getpWidth() {
        return pWidth;
    }

    public void setpWidth(int pWidth) {
        this.pWidth = pWidth;
    }

    public int getpHeight() {
        return pHeight;
    }

    public void setpHeight(int pHeight) {
        this.pHeight = pHeight;
    }

    public int getkNumber() {
        return kNumber;
    }

    public void setkNumber(int kNumber) {
        this.kNumber = kNumber;
    }
}

