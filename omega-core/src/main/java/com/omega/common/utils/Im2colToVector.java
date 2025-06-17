package com.omega.common.utils;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2colToVector extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5122995462148301836L;
    private int start = 0;
    private int end = 0;
    private float[][][][] x;
    private float[] y;
    private int kh;
    private int kw;
    private int stride;
    private int oHeight;
    private int oWidth;
    private int ow;
    private int kSize;

    public Im2colToVector(float[][][][] data, float[] y, int kh, int kw, int stride, int start, int end) {
        this.x = data;
        this.y = y;
        this.start = start;
        this.end = end;
        this.kh = kh;
        this.kw = kw;
        this.stride = stride;
        this.oHeight = ((x[0][0].length - kh) / stride) + 1;
        this.oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        this.ow = x[0].length * kh * kw;
        this.kSize = kh * kw;
    }

    public static float[] im2col(float[][][][] x, int kh, int kw, int stride) {
        int N = x.length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int ow = x[0].length * kh * kw;
        int oh = N * oHeight * oWidth;
        float[] result = new float[oh * ow];
        Im2colToVector job = new Im2colToVector(x, result, kh, kw, stride, 0, oh - 1);
        ForkJobEngine.run(job);
        return result;
    }

    public static void im2col(float[][][][] x, float[] y, int kh, int kw, int stride) {
        int N = x.length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int oh = N * oHeight * oWidth;
        Im2colToVector job = new Im2colToVector(x, y, kh, kw, stride, 0, oh - 1);
        ForkJobEngine.run(job);
    }

    public static float[] im2colKernel(float[][][][] x) {
        int ko = x.length;
        int kc = x[0].length;
        int kh = x[0][0].length;
        int kw = x[0][0][0].length;
        /**
         * kernel im2col

         */
        float[] col = new float[ko * kh * kw * kc];
        Im2colToVector job = new Im2colToVector(x, col, kh, kw, 0, 0, ko - 1);
        ForkJobEngine.run(job);
        return col;
    }

    public static float[] im2colV2(float[][][][] x, float[] result, int kh, int kw, int stride) {
        int N = x.length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int oh = N * oHeight * oWidth;
        Im2colToVector job = new Im2colToVector(x, result, kh, kw, stride, 0, oh - 1);
        ForkJobEngine.run(job);
        return result;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            col();
        } else {
            int mid = (start + end + 1) >>> 1;
            Im2colToVector left = new Im2colToVector(x, y, kh, kw, stride, start, mid - 1);
            Im2colToVector right = new Im2colToVector(x, y, kh, kw, stride, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void col() {
        for (int i = start; i <= end; i++) {
            int n = i / oHeight / oWidth;
            int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
            int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int c = j / kSize;
                int xSize = j - (c * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                y[i * ow + j] = x[n][c][xh][xw];
            }
        }
    }
}

