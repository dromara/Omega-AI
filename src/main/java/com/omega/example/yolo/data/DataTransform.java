package com.omega.example.yolo.data;

import com.omega.common.config.Tensor;

import java.util.Map;

public abstract class DataTransform {
    public abstract void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData);

    public abstract void showTransform(String outputPath, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData);
}

