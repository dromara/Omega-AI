package com.omega.engine.nn.layer.active.jobs.relu;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ReluBackwardJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5122995462148301836L;
    private int start = 0;
    private int end = 0;
    private float[] x;
    private float[] delta;
    private float[] diff;

    public ReluBackwardJob(float[] x, float[] delta, float[] diff, int start, int end) {
        this.x = x;
        this.delta = delta;
        this.diff = diff;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = end - start + 1;
        if (length < 8 || length <= delta.length / 8) {
            exeute();
        } else {
            int mid = (start + end + 1) >>> 1;
            ReluBackwardJob left = new ReluBackwardJob(x, delta, diff, start, mid - 1);
            ReluBackwardJob right = new ReluBackwardJob(x, delta, diff, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void exeute() {
        for (int i = start; i <= end; i++) {
            if (x[i] > 0) {
                diff[i] = delta[i];
            } else {
                diff[i] = 0;
            }
        }
    }
}

