package com.omega.engine.nn.data;

import com.omega.common.config.Tensor;
import com.omega.common.config.Tensors;

public abstract class BaseData {
    public int number = 0;
    public int channel = 0;
    public int height = 0;
    public int width = 0;
    public int labelSize = 0;
    public Tensor input;
    public Tensor label;
    public String[] labels;
    public String[] labelSet;

    public Tensor getRandomData(int[] indexs) {
        Tensor data = Tensors.tensor(indexs.length, channel, height, width);
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(this.input.data, index * channel * height * width, data.data, i * channel * height * width, channel * height * width);
        }
        return data;
    }

    public void getRandomData(int[] indexs, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(this.input.data, index * channel * height * width, input.data, i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.data, index * labelSize, label.data, i * labelSize, labelSize);
        }
    }

    public void getRandomData(int[] indexs, Tensor input) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(this.input.data, index * channel * height * width, input.data, i * channel * height * width, channel * height * width);
        }
    }

    public void randomData(int[] indexs, float[] data, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(data, index * channel * height * width, input.data, i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.data, index * labelSize, label.data, i * labelSize, labelSize);
        }
    }

    public void getAllData(Tensor input, Tensor label) {
        for (int i = 0; i < number; i++) {
            System.arraycopy(this.input.data, i * channel * height * width, input.data, i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.data, i * labelSize, label.data, i * labelSize, labelSize);
        }
    }

    public Tensor getOnceData(int index) {
        Tensor data = Tensors.tensor(1, channel, height, width);
        this.input.copy(index, data.data);
        return data;
    }

    public Tensor getOnceLabel(int index) {
        Tensor data = Tensors.tensor(1, label.channel, label.height, label.width);
        this.label.copy(index, data.data);
        return data;
    }

    public void getOnceData(int index, Tensor x) {
        this.input.copy(index, x.data);
        x.hostToDevice();
    }

    public void getBatchData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        if ((pageIndex + 1) * batchSize > this.number) {
            int input_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * channel * height * width;
            int label_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * labelSize;
            System.arraycopy(this.input.data, input_start, input.data, 0, batchSize * channel * height * width);
            System.arraycopy(this.label.data, label_start, label.data, 0, batchSize * labelSize);
        } else {
            int input_start = pageIndex * batchSize * channel * height * width;
            int label_start = pageIndex * batchSize * labelSize;
            System.arraycopy(this.input.data, input_start, input.data, 0, batchSize * channel * height * width);
            System.arraycopy(this.label.data, label_start, label.data, 0, batchSize * labelSize);
        }
    }

    public void getBatchData(int pageIndex, int batchSize, Tensor input) {
        if ((pageIndex + 1) * batchSize > this.number) {
            int input_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * channel * height * width;
            System.arraycopy(this.input.data, input_start, input.data, 0, batchSize * channel * height * width);
        } else {
            int input_start = pageIndex * batchSize * channel * height * width;
            System.arraycopy(this.input.data, input_start, input.data, 0, batchSize * channel * height * width);
        }
    }
}

