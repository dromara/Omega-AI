package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;

import java.util.Map;

/**
 * Updater
 *
 * @author Administrator
 */
public abstract class Updater {
    public Network net;
    public UpdaterType updaterType;
    public float beta = 0.9f;
    public float lr = 0.0001f;
    public Map<String, Float> params;
    public float[] vdw;
    public float[] vdgama;
    public float[] vdb;
    public float[] vdmw;  //c * kn * kh * kw
    public float[] vdmb;

    public abstract void update(Layer layer);

    public abstract void update(Layer layer, int batchSize);

    public abstract void updateForMatrix(Layer layer);

    public abstract void updateForBN(NormalizationLayer layer);

    public abstract UpdaterType getUpdaterType();
}

