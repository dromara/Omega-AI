package com.omega.engine.nn.network;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

public abstract class OutputsNetwork extends Network {
    public List<Layer> outputLayers = new ArrayList<Layer>();
    public int outputNum = 0;

    public abstract Tensor[] predicts(Tensor input);

    public abstract Tensor[] loss(Tensor label);

    public abstract Tensor[] lossDiff(Tensor label);

    public abstract void back(Tensor[] lossDiffs);

    public Tensor[] getOutputs() {
        Tensor[] outputs = new Tensor[outputNum];
        for (int i = 0; i < outputLayers.size(); i++) {
            outputs[i] = outputLayers.get(i).getOutput();
        }
        return outputs;
    }

    public void addLayer(Layer layer) {
        layer.setNetwork(this);
        layer.setIndex(this.layerList.size());
        layer.setUpdater(UpdaterFactory.create(this));
        if (layer.index <= 1) {
            layer.PROPAGATE_DOWN = false;
        }
        if (layer.isOutput) {
            this.outputNum++;
            this.outputLayers.add(layer);
        }
        this.layerList.add(layer);
    }

    public void setLossDiff(Tensor[] lossDiff) {
        for (int i = 0; i < outputLayers.size(); i++) {
            outputLayers.get(i).setDelta(lossDiff[i]);
        }
    }
}

