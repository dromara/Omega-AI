package com.omega.data.transformer.parallel;

import com.omega.engine.nn.network.Network;
import com.omega.data.transformer.JSONDatasetLoader;
import com.omega.data.transformer.parallel.params.DataLoaderParamters;

public abstract class ThreadDataset extends JSONDatasetLoader {
    private int batchSize = 1;

    public abstract void loadData(DataLoaderParamters params);

    public abstract DataLoaderParamters createParamters(Network network);

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}

