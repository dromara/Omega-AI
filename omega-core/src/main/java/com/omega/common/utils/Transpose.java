package com.omega.common.utils;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Transpose extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 3048313214839121739L;
    private int start = 0;
    private int end = 0;
    private float[] x;
    private float[] y;
    private int m;
    private int n;

    public Transpose(float[] x, float[] y, int m, int n, int start, int end) {
        this.x = x;
        this.y = y;
        this.m = m;
        this.n = n;
        this.start = start;
        this.end = end;
    }

    public static float[] transpose(float[] x, int m, int n) {
        float[] y = new float[m * n];
        Transpose job = new Transpose(x, y, m, n, 0, m * n - 1);
        ForkJobEngine.run(job);
        return y;
    }

    public static void transpose(float[] x, float[] y, int m, int n) {
        Transpose job = new Transpose(x, y, m, n, 0, m * n - 1);
        ForkJobEngine.run(job);
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            op();
        } else {
            int mid = (start + end + 1) >>> 1;
            Transpose left = new Transpose(x, y, m, n, start, mid - 1);
            Transpose right = new Transpose(x, y, m, n, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    public void op() {
        for (int i = start; i <= end; i++) {
            int r = i / n;
            int c = i % n;
            y[c * m + r] = x[r * n + c];
        }
    }
}

