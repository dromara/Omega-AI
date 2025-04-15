package com.omega.common.utils.kernal;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class KernalToVector2 extends RecursiveAction {
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
    private boolean T = false;

    public KernalToVector2(float[][][][] x, float[] y, boolean T, int start, int end) {
        this.x = x;
        this.y = y;
        this.T = T;
        this.H = x[0][0].length;
        this.W = x[0][0][0].length;
        this.N = x.length;
        this.C = x[0].length;
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
            KernalToVector2 left = new KernalToVector2(x, y, T, start, mid - 1);
            KernalToVector2 right = new KernalToVector2(x, y, T, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void execute() {
        for (int ko = start; ko <= end; ko++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        if (T) {
                            y[c * N * H * W + ko * H * W + h * W + w] = x[ko][c][h][w];
                        } else {
                            y[ko * C * H * W + c * H * W + h * W + w] = x[ko][c][h][w];
                        }
                    }
                }
            }
        }
    }
}

