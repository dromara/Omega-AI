package com.omega.data.yolo;

import com.omega.common.tensor.Tensor;

import java.util.Map;

public abstract class DataTransform {
    public abstract void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData);

    public abstract void showTransform(String outputPath, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData);
}

