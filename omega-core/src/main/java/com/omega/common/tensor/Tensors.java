package com.omega.common.tensor;

public class Tensors {
    public static Tensor tensor(int n, int c, int h, int w, float[] x, boolean isGPU) {
        return new Tensor(n, c, h, w, x, isGPU);
    }

    public static Tensor tensor(int n, int c, int h, int w, float[] x) {
        return new Tensor(n, c, h, w, x);
    }

    public static Tensor tensor(int n, int c, int h, int w) {
        return new Tensor(n, c, h, w);
    }

    public static Tensor tensor(float[][][][] x) {
        if (x != null) {
            int n = x.length;
            int c = x[0].length;
            int h = x[0][0].length;
            int w = x[0][0][0].length;
            return new Tensor(n, c, h, w);
        }
        return null;
    }
}

