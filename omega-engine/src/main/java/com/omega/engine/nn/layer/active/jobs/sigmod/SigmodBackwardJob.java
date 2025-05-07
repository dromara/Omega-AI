package com.omega.engine.nn.layer.active.jobs.sigmod;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class SigmodBackwardJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = -5122995462148301836L;
    private int start = 0;
    private int end = 0;
    private float[] delta;
    private float[] output;
    private float[] diff;

    public SigmodBackwardJob(float[] delta, float[] output, float[] diff, int start, int end) {
        this.output = output;
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
            SigmodBackwardJob left = new SigmodBackwardJob(delta, output, diff, start, mid - 1);
            SigmodBackwardJob right = new SigmodBackwardJob(delta, output, diff, mid, end);
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void exeute() {
        for (int i = start; i <= end; i++) {
            diff[i] = delta[i] * output[i] * (1.0f - output[i]);
        }
    }
}

