package com.omega.common.config;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NetworkConfig implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -8501799899931630742L;
    private String name;
    private String networkType;
    private List<LayerConfig> layers = new ArrayList<LayerConfig>();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getNetworkType() {
        return networkType;
    }

    public void setNetworkType(String networkType) {
        this.networkType = networkType;
    }

    public List<LayerConfig> getLayers() {
        return layers;
    }

    public void setLayers(List<LayerConfig> layers) {
        this.layers = layers;
    }
}

