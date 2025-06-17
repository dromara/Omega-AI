package com.omega.engine.database;

import com.omega.common.utils.MathUtils;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.tensor.Tensor;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataLoader {
    private DataSet trainData;
    private int batchSize;
    private float validSize = 0.2f;
    private List<Integer> trainIndex;
    private List<Integer> validIndex;
    private int channel;
    private int height;
    private int width;
    private int labelSize;

    public DataLoader(DataSet data, int batchSize, float validSize) {
        this.trainData = data;
        this.batchSize = batchSize;
        this.validSize = validSize;
        this.channel = data.channel;
        this.height = data.height;
        this.width = data.width;
        this.labelSize = data.labelSize;
        this.initDataset(validSize);
    }

    public void loadTrainData(int[] indexs, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(input.getData(), index * channel * height * width, input.getData(), i * channel * height * width, channel * height * width);
            System.arraycopy(label.getData(), index * labelSize, label.getData(), i * labelSize, labelSize);
        }
    }

    //
    //	public BaseData getTrainData(BaseData data) {
    //
    //
    //
    //	}
    //
    public int[][] getTrainIndex() {
        int itc = new BigDecimal(trainIndex.size()).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        int[][] tmp = new int[itc][batchSize];
        Collections.shuffle(trainIndex);
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= trainIndex.size()) {
                    tmp[i][j] = trainIndex.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = trainIndex.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    public int[][] getValidIndex() {
        int itc = new BigDecimal(validIndex.size()).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        int[][] tmp = new int[itc][batchSize];
        Collections.shuffle(validIndex);
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= validIndex.size()) {
                    tmp[i][j] = validIndex.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = validIndex.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    private void initDataset(float validSize) {
        try {
            int count = trainData.number;
            int trainCount = new BigDecimal(count).multiply(new BigDecimal(validSize)).ROUND_HALF_DOWN;
            int validCount = count - trainCount;
            Integer[] dataIndex = MathUtils.randomInts(count);
            trainIndex = new ArrayList<Integer>(trainCount);
            validIndex = new ArrayList<Integer>(validCount);
            loadDataset(dataIndex, trainIndex, validIndex);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    private void loadDataset(Integer[] dataIndex, List<Integer> trainIndex, List<Integer> validIndex) {
        for (int i = 0; i < trainIndex.size(); i++) {
            trainIndex.set(i, dataIndex[i]);
        }
        for (int i = 0; i < validIndex.size(); i++) {
            int index = i + (trainIndex.size() - 1);
            trainIndex.set(i, dataIndex[index]);
        }
    }
}

