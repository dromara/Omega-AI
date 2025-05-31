package com.omega.example.resnet.data;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class VailCodeFileDataLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static VailCodeFileDataLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private int[] indexs;
    private Tensor input;
    private List<String> keyList;
    private boolean normalization = false;

    public VailCodeFileDataLoader(String path, List<String> keyList, int[] indexs, int batchSize, Tensor input, boolean normalization, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.setIndexs(indexs);
        this.setInput(input);
        this.keyList = keyList;
        this.normalization = normalization;
    }

    public static VailCodeFileDataLoader getInstance(String path, List<String> keyList, int[] indexs, int batchSize, Tensor input, boolean normalization, int start, int end) {
        if (job == null) {
            job = new VailCodeFileDataLoader(path, keyList, indexs, batchSize, input, normalization, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
            }
            job.setPath(path);
            job.setStart(0);
            job.setEnd(end);
            job.setKeyList(keyList);
            job.setIndexs(indexs);
            job.reinitialize();
        }
        return job;
    }

    public static void load(String path, List<String> keyList, int[] indexs, int batchSize, Tensor input, boolean normalization) {
        VailCodeFileDataLoader job = getInstance(path, keyList, indexs, batchSize, input, normalization, 0, batchSize - 1);
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
            VailCodeFileDataLoader left = new VailCodeFileDataLoader(getPath(), keyList, getIndexs(), batchSize, input, normalization, getStart(), mid - 1);
            VailCodeFileDataLoader right = new VailCodeFileDataLoader(getPath(), keyList, getIndexs(), batchSize, input, normalization, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String filename = keyList.get(indexs[i]);
            String filePath = getPath() + "/" + filename;
            YoloImageUtils.loadImgDataToGrayTensor(filePath, input, i);
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

    public Tensor getInput() {
        return input;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    public List<String> getKeyList() {
        return keyList;
    }

    public void setKeyList(List<String> keyList) {
        this.keyList = keyList;
    }
}

