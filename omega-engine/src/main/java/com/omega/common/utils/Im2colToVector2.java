package com.omega.common.utils;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2colToVector2 extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5122995462148301836L;
    private int start = 0;
    private int end = 0;
    private float[][][][] x;
    private float[] y;
    private int N;
    private int kh;
    private int kw;
    private int stride;
    private int oHeight;
    private int oWidth;
    private int oh;
    private int ow;
    private int kSize;

    public Im2colToVector2(float[][][][] data, float[] y, int kh, int kw, int stride, int start, int end) {
        this.x = data;
        this.y = y;
        this.N = x.length;
        this.start = start;
        this.end = end;
        this.kh = kh;
        this.kw = kw;
        this.stride = stride;
        this.oHeight = ((x[0][0].length - kh) / stride) + 1;
        this.oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        this.oh = oHeight * oWidth;
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
        Im2colToVector2 job = new Im2colToVector2(x, result, kh, kw, stride, 0, oHeight * oWidth - 1);
        ForkJobEngine.run(job);
        return result;
    }

    public static void im2col(float[][][][] x, float[] y, int kh, int kw, int stride) {
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int oh = oHeight * oWidth;
        Im2colToVector2 job = new Im2colToVector2(x, y, kh, kw, stride, 0, oh - 1);
        ForkJobEngine.run(job);
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            col();
        } else {
            int mid = (start + end + 1) >>> 1;
            Im2colToVector2 left = new Im2colToVector2(x, y, kh, kw, stride, start, mid - 1);
            Im2colToVector2 right = new Im2colToVector2(x, y, kh, kw, stride, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void col() {
        for (int n = 0; n < N; n++) {
            for (int i = start; i <= end; i++) {
                int startH = i / oWidth * stride;
                int startW = i % oWidth * stride;
                for (int j = 0; j < ow; j++) {
                    int c = j / kSize;
                    int xSize = j - (c * kSize);
                    int xh = startH + xSize / kw;
                    int xw = startW + xSize % kw;
                    y[n * oh * ow + i * ow + j] = x[n][c][xh][xw];
                }
            }
        }
    }
}

