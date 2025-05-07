package com.omega.engine.nn.model;

import com.omega.engine.nn.network.Network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Network Init
 *
 * @author Administrator
 */
public class NetworkInit implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 608021449984902116L;
    private String networkType;
    private String lossType;
    private int channel = 0;
    private int height = 0;
    private int width = 0;
    private int oChannel = 0;
    private int oHeight = 0;
    private int oWidth = 0;
    private List<LayerInit> layers = new ArrayList<LayerInit>();

    public NetworkInit(Network network) {
        this.channel = network.getChannel();
        this.height = network.getHeight();
        this.width = network.getWidth();
        this.oChannel = network.oChannel;
        this.oHeight = network.oHeight;
        this.oWidth = network.oWidth;
        this.networkType = network.getNetworkType().getKey();
    }

    public String getNetworkType() {
        return networkType;
    }

    public void setNetworkType(String networkType) {
        this.networkType = networkType;
    }

    public List<LayerInit> getLayers() {
        return layers;
    }

    public void setLayers(List<LayerInit> layers) {
        this.layers = layers;
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

    public String getLossType() {
        return lossType;
    }

    public void setLossType(String lossType) {
        this.lossType = lossType;
    }
}

