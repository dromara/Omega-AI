package com.omega.engine.nn.layer.active;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.tensor.Tensor;

public abstract class ActiveFunctionLayer extends Layer {
    public Layer preLayer;

    public abstract void forward(Tensor input, int batch, int step);

    public abstract void back(Tensor delta, int batch, int step);

    public abstract void output(int batch, int step);

    public abstract void diff(int batch, int step);

    @Override
    public void init() {
        this.number = this.network.number;
        if (this.preLayer == null) {
            this.preLayer = this.network.getPreLayer(this.index);
            this.channel = preLayer.oChannel;
            this.height = preLayer.oHeight;
            this.width = preLayer.oWidth;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
        } else {
            this.number = preLayer.number;
        }
        if (output == null || number != output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if (this.preLayer == null) {
            this.preLayer = this.network.getPreLayer(this.index);
            this.channel = input.channel;
            this.height = input.height;
            this.width = input.width;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
        }
        if (output == null || number != output.number || input.channel != this.output.channel) {
            //			System.out.println("atc--->");
            this.output = Tensor.createGPUTensor(this.output, number, input.channel, input.height, input.width, true);
        }
    }

    @Override
    public void initBack() {
        //		System.out.println(this.index);
        if (this.diff == null) {
            this.diff = this.network.getDelta(this.index);
        }
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
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    }
}

