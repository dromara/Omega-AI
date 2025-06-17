package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class KernalToVector extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 479000399201982312L;
    private int start = 0;
    private int end = 0;
    private float[][][][] x;
    private float[] y;
    private int H = 0;
    private int W = 0;
    private int C = 0;
    private int N = 0;
    private boolean isT = true;

    public KernalToVector(float[][][][] x, float[] y, int start, int end, boolean isT) {
        this.x = x;
        this.y = y;
        this.H = x[0][0].length;
        this.W = x[0][0][0].length;
        this.N = x.length;
        this.C = x[0].length;
        this.start = start;
        this.end = end;
        this.isT = isT;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            execute();
        } else {
            int mid = (start + end + 1) >>> 1;
            KernalToVector left = new KernalToVector(x, y, start, mid - 1, isT);
            KernalToVector right = new KernalToVector(x, y, mid, end, isT);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void execute() {
        if (isT) {
            for (int i = start; i <= end; i++) {
                int n = i / H / W;
                int h = (i - (n * H * W)) / W;
                int w = (i - (n * H * W)) % W;
                for (int c = 0; c < C; c++) {
                    y[i * C + c] = x[n][c][h][w];
                }
            }
        } else {
            for (int i = start; i <= end; i++) {
                int c = i / H / W;
                int h = (i - (c * H * W)) / W;
                int w = (i - (c * H * W)) % W;
                for (int n = 0; n < N; n++) {
                    y[i * N + n] = x[n][c][h][w];
                }
            }
        }
    }
}

