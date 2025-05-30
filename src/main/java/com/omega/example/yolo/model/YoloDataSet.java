package com.omega.example.yolo.model;

import com.omega.engine.nn.data.BaseData;
import com.omega.engine.tensor.Tensors;

/**
 * data set
 *
 * @author Administrator
 */
public class YoloDataSet extends BaseData {
    public int inputSize = 0;

    public YoloDataSet(int number, int channel, int height, int width, int labelSize) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.inputSize = number * channel * height * width;
        this.labelSize = labelSize;
        this.input = Tensors.tensor(number, channel, height, width);
        this.label = Tensors.tensor(number, 1, 1, labelSize);
    }
}

