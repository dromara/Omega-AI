package com.omega.utils;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class OP1dto4d extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 1006668212815039650L;
    private int start = 0;
    private int end = 0;
    private float[] x;
    private float[][][][] y;
    private int N;
    private int C;
    private int H;
    private int W;
    private int offset = 0;

    public OP1dto4d(float[] data, float[][][][] col, int N, int C, int H, int W, int start, int end) {
        this.x = data;
        this.y = col;
        this.start = start;
        this.end = end;
        this.N = N;
        this.C = C;
        this.H = H;
        this.W = W;
        if (this.H != this.y[0][0].length) {
            this.offset = this.H - this.y[0][0].length;
            //			System.out.println("in");
        }
    }

    public static void to1d(float[] data, float[][][][] col, int N, int C, int H, int W) {
        OP1dto4d job = new OP1dto4d(data, col, N, C, H, W, 0, N * H * W - 1);
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
            OP1dto4d left = new OP1dto4d(x, y, N, C, H, W, start, mid - 1);
            OP1dto4d right = new OP1dto4d(x, y, N, C, H, W, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void col() {
        for (int i = start; i <= end; i++) {
            int n = i / H / W;
            int h = (i - (n * H * W)) / H;
            int w = (i - (n * H * W)) % H;
            for (int c = 0; c < C; c++) {
                int index = i * C + c;
                if (h - offset >= 0 && w - offset >= 0) {
                    this.y[n][c][h - offset][w - offset] = x[index];
                }
            }
        }
    }
}

