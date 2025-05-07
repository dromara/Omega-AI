package com.omega.example.yolo.data;

import com.omega.common.data.Tensor;

public abstract class BaseDataLoader {
    public int number;
    public int batchSize;
    public int labelSize;
    public int labelChannel;
    public String[] labelSet;

    public abstract int[][] shuffle();

    public abstract void loadData(int[] indexs, Tensor input);

    public abstract void loadData(int[] indexs, Tensor input, Tensor label);

    public abstract void loadData(int pageIndex, int batchSize, Tensor input, Tensor label);

    public abstract float[] loadData(int index);

    public abstract Tensor initLabelTensor();
}

