package com.omega.engine.nn.layer.normalization;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.tensor.Tensor;

public abstract class NormalizationLayer extends Layer {
    public Layer preLayer;
    public Tensor gamma;
    public Tensor beta;
    public Tensor runingMean;
    public Tensor runingVar;
    public Tensor diffGamma;
    public Tensor diffBeta;

    @Override
    public void init() {
        if (preLayer == null) {
            preLayer = this.network.getPreLayer(this.index);
            this.channel = preLayer.oChannel;
            this.height = preLayer.oHeight;
            this.width = preLayer.oWidth;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
        }
        this.number = this.network.number;
        initParam();
    }

    public void setPreLayer(Layer pre) {
        this.preLayer = pre;
        this.network = pre.network;
        this.channel = preLayer.oChannel;
        this.height = preLayer.oHeight;
        this.width = preLayer.oWidth;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
        network.paramLayers.add(this);
    }

    @Override
    public void initBack() {
    }

    public void getGradNorm() {
        if (network.CLIP_GRAD_NORM && diffGamma != null) {
            network.getGradNorm(this);
        }
    }
}

