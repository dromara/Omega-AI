package com.omega.engine.nn.data;

public class Matrix {
    public double[] data;
    public int[] shape;

    public Matrix(int[] shape) {
        this.shape = shape;
        this.initData();
    }

    public Matrix(double[] data, int[] shape) {
        this.data = data;
        this.shape = shape;
    }

    public void initData() {
        data = new double[shape[0] * shape[1] * shape[2] * shape[3]];
    }

    public Matrix clone() {
        return new Matrix(data, shape);
    }
}

