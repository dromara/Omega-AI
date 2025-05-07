package com.omega.engine.data;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;

import java.util.LinkedHashMap;

public class ParamsLoader {
    private int index;

    public void loadParams(Layer layer, LinkedHashMap<String, Object> params) {
        switch (layer.layerType) {
            case full:
                break;
            case conv:
                break;
            case bn:
                break;
        }
    }

    public void loadParamsToFull(FullyLayer layer, LinkedHashMap<String, Object> params) {
        //		params.
    }
}

