package com.omega.utils.image;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2colForWeight extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 1629952874768790706L;
    private int start = 0;
    private int end = 0;
    private float[][][][] x;
    private float[] y;
    private int kh;
    private int kw;
    private int stride;
    private int oHeight;
    private int oWidth;
    //	private int oh;
    private int ow;
    private int kSize;

    public Im2colForWeight(float[][][][] data, float[] y, int kh, int kw, int stride, int start, int end) {
        this.x = data;
        this.y = y;
        this.start = start;
        this.end = end;
        this.kh = kh;
        this.kw = kw;
        this.stride = stride;
        this.oHeight = ((x[0][0].length - kh) / stride) + 1;
        this.oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        this.ow = x.length * kh * kw;
        this.kSize = kh * kw;
    }

    public static void im2col(float[][][][] x, float[] y, int kh, int kw, int stride) {
        int C = x[0].length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int oh = C * oHeight * oWidth;
        Im2colForWeight job = new Im2colForWeight(x, y, kh, kw, stride, 0, oh - 1);
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
            Im2colForWeight left = new Im2colForWeight(x, y, kh, kw, stride, start, mid - 1);
            Im2colForWeight right = new Im2colForWeight(x, y, kh, kw, stride, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void col() {
        for (int i = start; i <= end; i++) {
            int c = i / oHeight / oWidth;
            int startH = (i - (c * oHeight * oWidth)) / oHeight * stride;
            int startW = (i - (c * oHeight * oWidth)) % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int n = j / kSize;
                int xSize = j - (n * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                y[i * ow + j] = x[n][c][xh][xw];
            }
        }
    }
}

