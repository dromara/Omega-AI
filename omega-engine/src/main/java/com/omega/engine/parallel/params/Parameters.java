package com.omega.engine.parallel.params;

import com.omega.engine.loss.LossType;
import com.omega.engine.updater.UpdaterType;

import java.io.Serializable;

public abstract class Parameters implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 5679315261220734693L;
    public LossType lossType;
    public UpdaterType updater;
    public float learnRate = 0.01f;
}

