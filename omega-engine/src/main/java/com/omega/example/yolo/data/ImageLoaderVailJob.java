package com.omega.example.yolo.data;

import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.OMImage;

import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ImageLoaderVailJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 7730441110162087756L;
    private static ImageLoaderVailJob job;
    private int start = 0;
    private int end = 0;
    private String path;
    private String extName;
    private Tensor input;
    private Tensor label;
    private String[] idxSet;
    private int[] indexs;
    private int boxes = 90;
    private int classes;
    private Map<String, float[]> orgLabelData;

    public ImageLoaderVailJob(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes, int start, int end) {
        this.setPath(path);
        this.setExtName(extName);
        this.setInput(input);
        this.setLabel(label);
        this.setIdxSet(idxSet);
        this.setIndexs(indexs);
        this.orgLabelData = orgLabelData;
        this.boxes = boxes;
        this.classes = classes;
        this.start = start;
        this.end = end;
    }

    public static ImageLoaderVailJob getInstance(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes, int start, int end) {
        if (job == null) {
            job = new ImageLoaderVailJob(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, start, end);
        } else {
            job.setPath(path);
            job.setExtName(extName);
            job.setIdxSet(idxSet);
            job.setIndexs(indexs);
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
            String key = getIdxSet()[indexs[i]];
            String imagePath = getPath() + "/" + key + "." + getExtName();
            OMImage orig = ImageLoader.loadImage(imagePath);
            float[] labelBoxs = this.orgLabelData.get(key);
            float[] labelXYWH = null;
            if (labelBoxs != null) {
                labelXYWH = ImageLoader.formatXYWH(labelBoxs, orig.getWidth(), orig.getHeight());
            }
            ImageLoader.loadVailDataDetection(input, label, i, orig, labelXYWH, input.width, input.height, boxes, classes);
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
            ImageLoaderVailJob left = new ImageLoaderVailJob(getPath(), getExtName(), input, label, getIdxSet(), indexs, orgLabelData, boxes, classes, getStart(), mid - 1);
            ImageLoaderVailJob right = new ImageLoaderVailJob(getPath(), getExtName(), input, label, getIdxSet(), indexs, orgLabelData, boxes, classes, mid, getEnd());
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

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public String getExtName() {
        return extName;
    }

    public void setExtName(String extName) {
        this.extName = extName;
    }

    public String[] getIdxSet() {
        return idxSet;
    }

    public void setIdxSet(String[] idxSet) {
        this.idxSet = idxSet;
    }
}

