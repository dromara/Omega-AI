package com.omega.engine.nn.data;

import com.omega.engine.tensor.Tensor;
import com.omega.engine.tensor.Tensors;

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
            System.arraycopy(this.input.getData(), index * channel * height * width, data.getData(), i * channel * height * width, channel * height * width);
        }
        return data;
    }

    public void getRandomData(int[] indexs, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(this.input.getData(), index * channel * height * width, input.getData(), i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.getData(), index * labelSize, label.getData(), i * labelSize, labelSize);
        }
    }

    public void getRandomData(int[] indexs, Tensor input) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(this.input.getData(), index * channel * height * width, input.getData(), i * channel * height * width, channel * height * width);
        }
    }

    public void randomData(int[] indexs, float[] data, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            System.arraycopy(data, index * channel * height * width, input.getData(), i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.getData(), index * labelSize, label.getData(), i * labelSize, labelSize);
        }
    }

    public void getAllData(Tensor input, Tensor label) {
        for (int i = 0; i < number; i++) {
            System.arraycopy(this.input.getData(), i * channel * height * width, input.getData(), i * channel * height * width, channel * height * width);
            System.arraycopy(this.label.getData(), i * labelSize, label.getData(), i * labelSize, labelSize);
        }
    }

    public Tensor getOnceData(int index) {
        Tensor data = Tensors.tensor(1, channel, height, width);
        this.input.copy(index, data.getData());
        return data;
    }

    public Tensor getOnceLabel(int index) {
        Tensor data = Tensors.tensor(1, label.getShape()[1], label.getShape()[2], label.getShape()[3]);
        this.label.copy(index, data.getData());
        return data;
    }

    public void getOnceData(int index, Tensor x) {
        this.input.copy(index, x.getData());
        x.hostToDevice();
    }

    public void getBatchData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        if ((pageIndex + 1) * batchSize > this.number) {
            int input_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * channel * height * width;
            int label_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * labelSize;
            System.arraycopy(this.input.getData(), input_start, input.getData(), 0, batchSize * channel * height * width);
            System.arraycopy(this.label.getData(), label_start, label.getData(), 0, batchSize * labelSize);
        } else {
            int input_start = pageIndex * batchSize * channel * height * width;
            int label_start = pageIndex * batchSize * labelSize;
            System.arraycopy(this.input.getData(), input_start, input.getData(), 0, batchSize * channel * height * width);
            System.arraycopy(this.label.getData(), label_start, label.getData(), 0, batchSize * labelSize);
        }
    }

    public void getBatchData(int pageIndex, int batchSize, Tensor input) {
        if ((pageIndex + 1) * batchSize > this.number) {
            int input_start = ((pageIndex) * batchSize - (batchSize - this.number % batchSize)) * channel * height * width;
            System.arraycopy(this.input.getData(), input_start, input.getData(), 0, batchSize * channel * height * width);
        } else {
            int input_start = pageIndex * batchSize * channel * height * width;
            System.arraycopy(this.input.getData(), input_start, input.getData(), 0, batchSize * channel * height * width);
        }
    }
}

