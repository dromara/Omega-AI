package com.omega.engine.nn.model;

import com.omega.common.config.Tensor;
import com.omega.engine.nn.layer.Layer;

import java.io.Serializable;

/**
 * LayerInit
 *
 * @author Administrator
 */
public class LayerInit implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -438190372990903311L;
    private String layerType;
    private int index = 0;
    private String updater;
    private int channel = 0;
    private int height = 0;
    private int width = 0;
    private int oChannel = 0;
    private int oHeight = 0;
    private int oWidth = 0;
    private Tensor weight;
    private Tensor bias;

    public LayerInit(Layer layer) {
        this.index = layer.index;
        this.channel = layer.channel;
        this.height = layer.height;
        this.width = layer.width;
        this.oChannel = layer.oChannel;
        this.oHeight = layer.oHeight;
        this.oWidth = layer.oWidth;
        this.weight = layer.weight;
        this.bias = layer.bias;
        this.layerType = layer.getLayerType().getKey();
        if (layer.updater != null) {
            this.updater = layer.updater.getUpdaterType().getKey();
        }
    }

    public String getLayerType() {
        return layerType;
    }

    public void setLayerType(String layerType) {
        this.layerType = layerType;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public String getUpdater() {
        return updater;
    }

    public void setUpdater(String updater) {
        this.updater = updater;
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
}

