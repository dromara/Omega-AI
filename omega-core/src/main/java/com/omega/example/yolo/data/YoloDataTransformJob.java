package com.omega.example.yolo.data;

import com.omega.common.data.utils.DataTransforms;
import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;

import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class YoloDataTransformJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 7096469376344662548L;
    private static YoloDataTransformJob job;
    private int start = 0;
    private int end = 0;
    private float jitter = 0.1f;
    private float hue = 0.1f;
    private float saturation = 0.75f;
    private float exposure = 0.75f;
    private int classnum = 1;
    private int numBoxes = 7;
    private DataType dataType;
    private Tensor input;
    private Tensor label;
    private String[] idxSet;
    private int[] indexs;
    private Map<String, float[]> orgLabelData;

    public YoloDataTransformJob(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, DataType dataType, int numBoxes, int start, int end) {
        this.setInput(input);
        this.setLabel(label);
        this.idxSet = idxSet;
        this.setIndexs(indexs);
        this.orgLabelData = orgLabelData;
        this.dataType = dataType;
        this.start = start;
        this.end = end;
        this.numBoxes = numBoxes;
    }

    public static YoloDataTransformJob getInstance(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, DataType dataType, int numBoxes, int start, int end) {
        if (job == null) {
            job = new YoloDataTransformJob(input, label, idxSet, indexs, orgLabelData, dataType, numBoxes, start, end);
        } else {
            job.setIndexs(indexs);
            job.setInput(input);
            job.setLabel(label);
            job.setStart(0);
            job.setEnd(end);
            job.reinitialize();
        }
        return job;
    }

    public static void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, DataType dataType, int numBoxes) {
        YoloDataTransformJob job = YoloDataTransformJob.getInstance(input, label, idxSet, indexs, orgLabelData, dataType, numBoxes, 0, input.getShape()[0] - 1);
        ForkJobEngine.run(job);
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= input.getShape()[0] / 8) {
            transform();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            YoloDataTransformJob left = new YoloDataTransformJob(getInput(), getLabel(), idxSet, getIndexs(), orgLabelData, dataType, numBoxes, getStart(), mid - 1);
            YoloDataTransformJob right = new YoloDataTransformJob(getInput(), getLabel(), idxSet, getIndexs(), orgLabelData, dataType, numBoxes, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    public void transform() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String key = idxSet[getIndexs()[i]];
            float[] orgList = this.orgLabelData.get(key);
            int labelSize = orgList.length / 5;
            float[] rLabel = new float[orgList.length];
            float[] img = new float[input.getOnceSize()];
            //			rLabel = this.orgLabelData.get(key);
            /**
             * 随机裁剪边缘

             */
            DataTransforms.randomCropWithLabel(i, getInput(), img, orgList, rLabel, labelSize, getInput().getShape()[2], getInput().getShape()[3], jitter);
            /**
             * 随机上下反转

             */
            DataTransforms.randomHorizontalFilpWithLabel(getInput(), img, i, rLabel, rLabel, labelSize);
            /**
             * hsv变换

             */
            getInput().getByNumber(i, img);
            YoloDataTransform.randomDistortImage(img, getInput().getShape()[3], getInput().getShape()[2], getInput().getShape()[1], hue, saturation, exposure);
            getInput().setByNumber(i, img);
            /**
             * 转换对应版本yolo label

             */
            switch (dataType) {
                case yolov1:
                    YoloDataTransform.loadLabelToYolov1(rLabel, i, getLabel(), input.getShape()[2], input.getShape()[3], classnum, this.numBoxes);
                    break;
                case yolov3:
                    YoloDataTransform.loadLabelToYolov3(rLabel, i, getLabel(), input.getShape()[2], input.getShape()[3], this.numBoxes);
                    break;
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

    public Tensor getInput() {
        return input;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    public Tensor getLabel() {
        return label;
    }

    public void setLabel(Tensor label) {
        this.label = label;
    }

    public int getNumBoxes() {
        return numBoxes;
    }

    public void setNumBoxes(int numBoxes) {
        this.numBoxes = numBoxes;
    }
}

