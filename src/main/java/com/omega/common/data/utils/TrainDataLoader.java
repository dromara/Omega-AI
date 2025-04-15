package com.omega.common.config.utils;

import com.omega.common.utils.MathUtils;
import com.omega.engine.nn.data.DataSet;

import java.math.BigDecimal;

public class TrainDataLoader {
    public static float testRate = 0.2f;

    public static DataSet[] loadData(DataSet dataSet, float rate) {
        DataSet[] set = new DataSet[2];
        try {
            Integer[] indexs = MathUtils.randomInts(dataSet.number);
            int split = new BigDecimal(rate).multiply(new BigDecimal(dataSet.number)).intValue();
            Integer[] train_idx = new Integer[dataSet.number - split];
            Integer[] vail_idx = new Integer[split];
            System.arraycopy(indexs, 0, train_idx, 0, dataSet.number - split);
            System.arraycopy(indexs, split - 1, vail_idx, 0, split);
            DataSet trainSet = new DataSet(dataSet.number - split, dataSet.channel, dataSet.height, dataSet.width, dataSet.labelSize, dataSet.labelSet);
            DataSet vailSet = new DataSet(split, dataSet.channel, dataSet.height, dataSet.width, dataSet.labelSize, dataSet.labelSet);
            loadDataByIdx(train_idx, dataSet, trainSet);
            loadDataByIdx(vail_idx, dataSet, vailSet);
            set[0] = trainSet;
            set[1] = vailSet;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return set;
    }

    private static void loadDataByIdx(Integer[] idx, DataSet org, DataSet dest) {
        int channel = org.channel;
        int width = org.width;
        int height = org.height;
        int labelSize = org.labelSize;
        for (int i = 0; i < idx.length; i++) {
            int index = idx[i];
            System.arraycopy(org.input.data, index * channel * height * width, dest.input.data, i * channel * height * width, channel * height * width);
            System.arraycopy(org.label.data, index * labelSize, dest.label.data, i * labelSize, labelSize);
            System.arraycopy(org.labels, index, dest.labels, i, 1);
        }
    }
}

