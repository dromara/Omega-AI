package com.omega.common.utils;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.data.DataSet;
import com.omega.example.yolo.data.ImageLoader;

public abstract class DataloarderTransforms {
    public static DataSet resize(DataSet org, int th, int tw) {
        Tensor oi = org.input;
        float[] target = new float[org.number * org.channel * th * tw];
        for (int b = 0; b < oi.number; b++) {
            float[] sized = ImageLoader.resized(oi.getByNumber(b), oi.channel, oi.width, oi.height, oi.channel, tw, th);
            System.arraycopy(sized, 0, target, b * sized.length, sized.length);
        }
        oi.data = target;
        oi.setHeight(th);
        oi.setWidth(tw);
        org.height = th;
        org.width = tw;
        return org;
    }

    public static DataSet normalize(DataSet org, float[] mean, float[] std) {
        int oc = org.channel;
        int oh = org.height;
        int ow = org.width;
        for (int b = 0; b < org.number; b++) {
            for (int c = 0; c < oc; c++) {
                for (int h = 0; h < oh; h++) {
                    for (int w = 0; w < ow; w++) {
                        float val = org.input.data[b * oc * oh * ow + c * oh * ow + h * ow + w];
                        org.input.data[b * oc * oh * ow + c * oh * ow + h * ow + w] = ((val / 255.0f) - mean[c]) / std[c];
                    }
                }
            }
        }
        return org;
    }

    public abstract void compose(DataSet org);
}

