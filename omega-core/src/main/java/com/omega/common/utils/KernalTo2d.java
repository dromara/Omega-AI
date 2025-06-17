package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class KernalTo2d extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6036079144782573143L;
    private int start = 0;
    private int end = 0;
    private float[][][][] x;
    private float[][] y;
    private int H = 0;
    private int W = 0;

    public KernalTo2d(float[][][][] x, float[][] y, int start, int end) {
        this.x = x;
        this.y = y;
        this.H = x[0][0].length;
        this.W = x[0][0][0].length;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            execute();
        } else {
            int mid = (start + end + 1) >>> 1;
            KernalTo2d left = new KernalTo2d(x, y, start, mid - 1);
            KernalTo2d right = new KernalTo2d(x, y, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void execute() {
        for (int i = start; i <= end; i++) {
            int n = i / H / W;
            int h = (i - (n * H * W)) / W;
            int w = (i - (n * H * W)) % W;
            for (int c = 0; c < y[i].length; c++) {
                y[i][c] = x[n][c][h][w];
            }
        }
    }
}

