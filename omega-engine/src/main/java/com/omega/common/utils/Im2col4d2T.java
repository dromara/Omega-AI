package com.omega.common.utils;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2col4d2T extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5701105089303774359L;
    private int start = 0;
    private int end = 0;
    private float[] x;
    private float[][][][] y;
    private int kh;
    private int kw;
    private int kc;
    private int kn;
    private float number;

    public Im2col4d2T(float[] data, float[][][][] col, int kn, int kc, int kh, int kw, int start, int end, float number) {
        this.x = data;
        this.y = col;
        this.start = start;
        this.end = end;
        this.kh = kh;
        this.kw = kw;
        this.kc = kc;
        this.kn = kn;
        this.number = number;
    }

    public static void to4d(float[] data, float[][][][] col, int kn, int kc, int kh, int kw, float number) {
        Im2col4d2T job = new Im2col4d2T(data, col, kn, kc, kh, kw, 0, kc * kh * kw - 1, number);
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
            Im2col4d2T left = new Im2col4d2T(x, y, kn, kc, kh, kw, start, mid - 1, number);
            Im2col4d2T right = new Im2col4d2T(x, y, kn, kc, kh, kw, mid, end, number);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void col() {
        for (int i = start; i <= end; i++) {
            int ci = i / kh / kw;
            int ckh = (i - (ci * kh * kw)) / kh;
            int ckw = (i - (ci * kh * kw)) % kh;
            for (int o = 0; o < kn; o++) {
                int index = i * kn + o;
                if (ckh < this.y[o][ci].length && ckw < this.y[o][ci][0].length) {
                    if (number > 0) {
                        this.y[o][ci][ckh][ckw] = x[index] / number;
                    } else {
                        this.y[o][ci][ckh][ckw] = x[index];
                    }
                }
            }
        }
    }
}

