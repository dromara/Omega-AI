package com.omega.engine.nn.layer.utils;

import com.omega.engine.nn.layer.Layer;

public abstract class LayerHook {
    public abstract void runHookFn(Layer layer);
}

