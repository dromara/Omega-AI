package com.omega.common.utils;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class AddDemo extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private float[] x;
    private float[] y;
    private float[] result;
    private int start, end;

    public AddDemo(float[] x, float[] y, float[] result, int start, int end) {
        // TODO Auto-generated constructor stub
        this.x = x;
        this.y = y;
        this.result = result;
        this.start = start;
        this.end = end;
    }

    public static void testIm2colInput() {
        ForkJoinPool forkJoinPool = new ForkJoinPool();
        int n = 100000000;
        float[] x = new float[n];
        float[] y = new float[n];
        float[] result = new float[n];
        AddDemo im2col = new AddDemo(x, y, result, 0, n - 1);
        ForkJoinTask<Void> a = forkJoinPool.submit(im2col);
        long start = System.nanoTime();
        a.join();
        System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
    }

    public static void main(String[] args) {
        AddDemo.testIm2colInput();
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            toCol();
        } else {
            int mid = (start + end + 1) >>> 1;
            AddDemo left = new AddDemo(x, y, result, start, mid - 1);
            AddDemo right = new AddDemo(x, y, result, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void toCol() {
        for (int n = start; n <= end; n++) {
            result[n] = x[n] + y[n];
        }
    }
}

