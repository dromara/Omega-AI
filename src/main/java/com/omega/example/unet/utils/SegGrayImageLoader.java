package com.omega.example.unet.utils;

import com.omega.common.config.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class SegGrayImageLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static SegGrayImageLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private String[] names;
    private int[] indexs;
    private Tensor input;
    private String extName;
    private boolean normalization = false;
    private boolean gray = false;

    public SegGrayImageLoader(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, boolean gray, boolean normalization, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.extName = extName;
        this.setNames(names);
        this.setIndexs(indexs);
        this.setInput(input);
        this.gray = gray;
        this.normalization = normalization;
    }

    public static SegGrayImageLoader getInstance(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, boolean gray, boolean normalization, int start, int end) {
        if (job == null) {
            job = new SegGrayImageLoader(path, extName, names, indexs, batchSize, input, gray, normalization, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
            }
            job.setPath(path);
            job.setNames(names);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.reinitialize();
        }
        return job;
    }

    public static void load(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, boolean gray, boolean normalization) {
        //		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
        SegGrayImageLoader job = getInstance(path, extName, names, indexs, batchSize, input, gray, normalization, 0, batchSize - 1);
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
            SegGrayImageLoader left = new SegGrayImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), gray, normalization, getStart(), mid - 1);
            SegGrayImageLoader right = new SegGrayImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), gray, normalization, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String filePath = getPath() + "/" + getNames()[getIndexs()[i]];
            if (!getNames()[getIndexs()[i]].contains(".")) {
                filePath = getPath() + "/" + getNames()[getIndexs()[i]] + "." + extName;
            }
            if (gray) {
                float[] data = YoloImageUtils.loadImgDataToGrayArray(filePath, normalization);
                System.arraycopy(data, 0, getInput().data, i * getInput().channel * getInput().height * getInput().width, getInput().channel * getInput().height * getInput().width);
            } else {
                float[] data = YoloImageUtils.loadImgDataToArray(filePath, normalization);
                System.arraycopy(data, 0, getInput().data, i * getInput().channel * getInput().height * getInput().width, getInput().channel * getInput().height * getInput().width);
            }
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
}

