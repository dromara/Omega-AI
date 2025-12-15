package com.omega.example.unet.utils;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class SegImageLoader2 extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static SegImageLoader2 job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private String path2;
    private String[] names;
    private int[] indexs;
    private Tensor input;
    private Tensor input2;
    private String extName;
    private boolean normalization = false;
    private boolean gray = false;
    private float[] mean;
    private float[] std;
    private float[] mean2;
    private float[] std2;

    public SegImageLoader2(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.setPath2(path2);
        this.extName = extName;
        this.setNames(names);
        this.setIndexs(indexs);
        this.setInput(input);
        this.setInput2(input2);
        this.gray = gray;
        this.normalization = normalization;
    }

    public SegImageLoader2(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization, float[] mean, float[] std, float[] mean2, float[] std2, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.setPath2(path2);
        this.extName = extName;
        this.setNames(names);
        this.setIndexs(indexs);
        this.setInput(input);
        this.setInput2(input2);
        this.mean = mean;
        this.std = std;
        this.mean2 = mean2;
        this.std2 = std2;
        this.gray = gray;
        this.normalization = normalization;
    }

    public static SegImageLoader2 getInstance(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization, int start, int end) {
        if (job == null) {
            job = new SegImageLoader2(path, path2, extName, names, indexs, batchSize, input, input2, gray, normalization, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
                job.setInput2(input2);
            }
            job.setPath(path);
            job.setPath2(path2);
            job.setNames(names);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.reinitialize();
        }
        return job;
    }

    public static SegImageLoader2 getInstance(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization, float[] mean, float[] std, float[] mean2, float[] std2, int start, int end) {
        if (job == null) {
            job = new SegImageLoader2(path, path2, extName, names, indexs, batchSize, input, input2, gray, normalization, mean, std, mean2, std2, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
                job.setInput2(input2);
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

    public static void load(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization) {
        //		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
        SegImageLoader2 job = getInstance(path, path2, extName, names, indexs, batchSize, input, input2, gray, normalization, 0, batchSize - 1);
        ForkJobEngine.run(job);
    }

    public static void load(String path, String path2, String extName, String[] names, int[] indexs, int batchSize, Tensor input, Tensor input2, boolean gray, boolean normalization, float[] mean, float[] std, float[] mean2, float[] std2) {
        SegImageLoader2 job = getInstance(path, path2, extName, names, indexs, batchSize, input, input2, gray, normalization, mean, std, mean2, std2, 0, batchSize - 1);
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
            SegImageLoader2 left = null;
            SegImageLoader2 right = null;
            if (mean != null) {
                left = new SegImageLoader2(getPath(), path2, extName, getNames(), getIndexs(), batchSize, getInput(), input2, gray, normalization, mean, std, mean2, std2, getStart(), mid - 1);
                right = new SegImageLoader2(getPath(), path2, extName, getNames(), getIndexs(), batchSize, getInput(), input2, gray, normalization, mean, std, mean2, std2, mid, getEnd());
            } else {
                left = new SegImageLoader2(getPath(), path2, extName, getNames(), getIndexs(), batchSize, getInput(), input2, gray, normalization, getStart(), mid - 1);
                right = new SegImageLoader2(getPath(), path2, extName, getNames(), getIndexs(), batchSize, getInput(), input2, gray, normalization, mid, getEnd());
            }
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        	int idx = getIndexs()[i];
            String filePath = getPath() + "/" + getNames()[idx];
            String filePath2 = getPath2() + "/" + getNames()[idx];
            if (!getNames()[getIndexs()[i]].contains(".")) {
                filePath = getPath() + "/" + getNames()[idx] + "." + extName;
                filePath2 = getPath2() + "/" + getNames()[idx] + "." + extName;
            }
            if (gray) {
                float[] data = YoloImageUtils.loadImgDataToGrayArray(filePath, normalization);
                System.arraycopy(data, 0, getInput().data, i * getInput().channel * getInput().height * getInput().width, getInput().channel * getInput().height * getInput().width);
                data = YoloImageUtils.loadImgDataToGrayArray(filePath2, normalization);
                System.arraycopy(data, 0, getInput2().data, i * getInput2().channel * getInput2().height * getInput2().width, getInput2().channel * getInput2().height * getInput2().width);
            } else {
                float[] data = null;
                if (mean != null) {
                    data = YoloImageUtils.loadImgDataToArray(filePath, normalization, mean, std);
                } else {
                    data = YoloImageUtils.loadImgDataToArray(filePath, normalization);
                }
//                				System.out.println(filePath+data);
                System.arraycopy(data, 0, getInput().data, i * getInput().channel * getInput().height * getInput().width, getInput().channel * getInput().height * getInput().width);

                if (mean2 != null) {
                    data = YoloImageUtils.loadImgDataToArray(filePath2, normalization, mean2, std2);
                } else {
                    data = YoloImageUtils.loadImgDataToArray(filePath2, normalization);
                }
//                				System.out.println(filePath+data);
                System.arraycopy(data, 0, getInput2().data, i * getInput2().channel * getInput2().height * getInput2().width, getInput2().channel * getInput2().height * getInput2().width);
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

	public Tensor getInput2() {
		return input2;
	}

	public void setInput2(Tensor input2) {
		this.input2 = input2;
	}

	public String getPath2() {
		return path2;
	}

	public void setPath2(String path2) {
		this.path2 = path2;
	}
}

