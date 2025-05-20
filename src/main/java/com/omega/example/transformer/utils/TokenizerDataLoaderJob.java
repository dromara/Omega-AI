package com.omega.example.transformer.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.engine.tensor.Tensor;

public class TokenizerDataLoaderJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 7730441110162087756L;
    private static TokenizerDataLoaderJob job;
    private int start = 0;
    private int end = 0;
    private String path;
    private Tensor input;
    private Tensor label;
    private int[] indexs;

    public TokenizerDataLoaderJob(String path, Tensor input, Tensor label, int start, int end) {
        this.path = path;
        this.setInput(input);
        this.setLabel(label);
        this.setIndexs(indexs);
        this.start = start;
        this.end = end;
    }

    public static TokenizerDataLoaderJob getInstance(String path, Tensor input, Tensor label, int start, int end) {
        if (job == null) {
            job = new TokenizerDataLoaderJob(path, input, label, start, end);
        } else {
            job.setInput(input);
            job.setLabel(label);
            job.setStart(0);
            job.setEnd(end);
            job.reinitialize();
        }
        return job;
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        }
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= input.number / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            TokenizerDataLoaderJob left = new TokenizerDataLoaderJob(path, input, label, getStart(), mid - 1);
            TokenizerDataLoaderJob right = new TokenizerDataLoaderJob(path, input, label, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public void setIndexs(int[] indexs) {
        this.indexs = indexs;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    public void setLabel(Tensor label) {
        this.label = label;
    }
}

