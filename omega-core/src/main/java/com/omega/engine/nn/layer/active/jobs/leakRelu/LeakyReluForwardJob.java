package com.omega.engine.nn.layer.active.jobs.leakRelu;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class LeakyReluForwardJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5122995462148301836L;
    private int start = 0;
    private int end = 0;
    private float[] x;
    private float[] y;
    private float leak = 0.2f;

    public LeakyReluForwardJob(float[] x, float[] y, int start, int end) {
        this.x = x;
        this.y = y;
        this.start = start;
        this.end = end;
    }

    public void setX(float[] x) {
        this.x = x;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= x.length / 8) {
            exeute();
        } else {
            int mid = (start + end + 1) >>> 1;
            LeakyReluForwardJob left = new LeakyReluForwardJob(x, y, start, mid - 1);
            LeakyReluForwardJob right = new LeakyReluForwardJob(x, y, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void exeute() {
        for (int i = start; i <= end; i++) {
            if (x[i] > 0) {
                y[i] = x[i];
            } else {
                y[i] = x[i] * leak;
            }
        }
    }
}

