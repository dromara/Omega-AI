package com.omega.example.asr.utils;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * WAVFBankLoader
 *
 * @author Administrator
 */
public class WAVFBankLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static WAVFBankLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private String[] names;
    private int[] indexs;
    private Tensor input;
    private Tensor inputLen;
    private int num_bins;
    private int maxLen;

    public WAVFBankLoader(String path, String[] names, int[] indexs, int batchSize, Tensor input, Tensor inputLen, int num_bins, int maxLen, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.setNames(names);
        this.setIndexs(indexs);
        this.setInput(input);
        this.setInputLen(inputLen);
        this.setNum_bins(num_bins);
        this.setMaxLen(maxLen);
    }

    public static WAVFBankLoader getInstance(String path, String[] names, int[] indexs, int batchSize, Tensor input, Tensor inputLen, int num_bins, int maxLen, int start, int end) {
        if (job == null) {
            job = new WAVFBankLoader(path, names, indexs, batchSize, input, inputLen, num_bins, maxLen, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
                job.setInputLen(inputLen);
            }
            job.setPath(path);
            job.setNames(names);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.setNum_bins(num_bins);
            job.setMaxLen(maxLen);
            job.reinitialize();
        }
        return job;
    }

    public static void load(String path, String[] names, int[] indexs, int batchSize, Tensor input, Tensor inputLen, int num_bins, int maxLen) {
        WAVFBankLoader job = getInstance(path, names, indexs, batchSize, input, inputLen, num_bins, maxLen, 0, batchSize - 1);
        ForkJobEngine.run(job);
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= batchSize / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            WAVFBankLoader left = null;
            WAVFBankLoader right = null;
            left = new WAVFBankLoader(getPath(), getNames(), getIndexs(), batchSize, getInput(), getInputLen(), num_bins, maxLen, getStart(), mid - 1);
            right = new WAVFBankLoader(getPath(), getNames(), getIndexs(), batchSize, getInput(), getInputLen(), num_bins, maxLen, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String filePath = getPath() + "/" + getNames()[getIndexs()[i]];
            float[] data = FBank.fbank(filePath, getNum_bins(), maxLen, i, getInputLen().getData());
            System.arraycopy(data, 0, getInput().getData(), i * getInput().getShape()[1] * getInput().getShape()[2] * getInput().getShape()[3], getInput().getShape()[1] * getInput().getShape()[2] * getInput().getShape()[3]);
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

    public int[] getIndexs() {
        return indexs;
    }

    public void setIndexs(int[] indexs) {
        this.indexs = indexs;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public String[] getNames() {
        return names;
    }

    public void setNames(String[] names) {
        this.names = names;
    }

    public Tensor getInput() {
        return input;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    public int getNum_bins() {
        return num_bins;
    }

    public void setNum_bins(int num_bins) {
        this.num_bins = num_bins;
    }

    public int getMaxLen() {
        return maxLen;
    }

    public void setMaxLen(int maxLen) {
        this.maxLen = maxLen;
    }

    public Tensor getInputLen() {
        return inputLen;
    }

    public void setInputLen(Tensor inputLen) {
        this.inputLen = inputLen;
    }
}
