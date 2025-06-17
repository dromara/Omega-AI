package com.omega.example.yolo.data;

import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.OMImage;

import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ImageLoaderJob extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 7730441110162087756L;
    private static ImageLoaderJob job;
    private static LoaderMemCachePool pool = new LoaderMemCachePool();
    private int start = 0;
    private int end = 0;
    private String path;
    private String extName;
    private Tensor input;
    private Tensor label;
    private String[] idxSet;
    private int[] indexs;
    private float jitter = 0.1f;
    private float hue = 0.1f;
    private float saturation = 1.5f;
    private float exposure = 1.5f;
    private int classes = 2;
    private int boxes = 90;
    private float resized = 1.5f;
    private int letter_box = 1;
    private Map<String, float[]> orgLabelData;

    public ImageLoaderJob(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes, int start, int end) {
        this.path = path;
        this.extName = extName;
        this.setInput(input);
        this.setLabel(label);
        this.idxSet = idxSet;
        this.setIndexs(indexs);
        this.orgLabelData = orgLabelData;
        this.boxes = boxes;
        this.classes = classes;
        this.start = start;
        this.end = end;
    }

    public static ImageLoaderJob getInstance(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes, int start, int end) {
        if (job == null) {
            job = new ImageLoaderJob(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, start, end);
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

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String key = idxSet[indexs[i]];
            //			System.out.println(indexs[i]+":"+i+":"+key);
            float[] labelBoxs = this.orgLabelData.get(key);
            String imagePath = path + "/" + key;
            if(!key.contains(extName)) {
            	imagePath += "." + extName;
            }
            //			MemeryBlock block = pool.getBlock(4096 * 4096 * 3);
            //			OMImage orig = ImageLoader.loadImage(imagePath, block.getData());
            OMImage orig = ImageLoader.loadImage(imagePath);
//            System.out.println(imagePath);
            float[] labelXYWH = ImageLoader.formatXYWH(labelBoxs, orig.getWidth(), orig.getHeight());
            //			ImageLoader.loadDataDetection(input, label, i, orig, labelXYWH, input.width, input.height, boxes, classes, jitter, hue, saturation, exposure);
            ImageLoader.loadDataDetection2(input, label, i, orig, labelXYWH, input.getShape()[3], input.getShape()[2], boxes, classes, jitter, hue, saturation, exposure, resized, letter_box);
            //			block.free();
        }
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= input.getShape()[0] / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            ImageLoaderJob left = new ImageLoaderJob(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, getStart(), mid - 1);
            ImageLoaderJob right = new ImageLoaderJob(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, mid, getEnd());
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

