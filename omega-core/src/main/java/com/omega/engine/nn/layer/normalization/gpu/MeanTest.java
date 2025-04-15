package com.omega.engine.nn.layer.normalization.gpu;

public class MeanTest {
    public static float mean(float[] x) {
        float mean = 0.0f;
        for (int i = 0; i < x.length; i++) {
            mean += x[i];
        }
        return mean / x.length;
    }

    public static void main(String args[]) {
        float[] x = new float[]{1, 2, 3, 4, 4, 5, 6, 7, -10, 2, -9};
        float mean = mean(x);
        System.out.println(mean);
        float tmp = 0.0f;
        for (int i = 0; i < x.length; i++) {
            tmp += (x[i] - mean);
        }
        System.out.println(tmp / x.length);
    }
}

