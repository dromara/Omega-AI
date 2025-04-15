package com.omega.utils.diffusion;

import com.omega.common.tensor.Tensor;
import com.omega.common.utils.data.DataTransforms;
import com.omega.common.task.ForkJobEngine;
import com.omega.utils.yolo.YoloImageUtils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * DuffsionImageLoader
 *
 * @author Administrator
 */
public class DiffusionImageLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static DiffusionImageLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private String[] names;
    private int[] indexs;
    private Tensor input;
    private String extName;
    private boolean normalization = false;
    private boolean horizontalFilp = false;
    private float[] a;
    private float[] b;
    private float[] noise;
    private float[] mean = new float[]{0.5f, 0.5f, 0.5f};
    private float[] std = new float[]{0.5f, 0.5f, 0.5f};

    public DiffusionImageLoader(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, float[] a, float[] b, float[] noise, boolean normalization, boolean horizontalFilp, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setPath(path);
        this.extName = extName;
        this.setNames(names);
        this.setIndexs(indexs);
        this.setInput(input);
        this.setA(a);
        this.setB(b);
        this.setNoise(noise);
        this.horizontalFilp = horizontalFilp;
        this.normalization = normalization;
    }

    public static DiffusionImageLoader getInstance(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, float[] a, float[] b, float[] noise, boolean normalization, boolean horizontalFilp, int start, int end) {
        if (job == null) {
            job = new DiffusionImageLoader(path, extName, names, indexs, batchSize, input, a, b, noise, normalization, horizontalFilp, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
            }
            job.setPath(path);
            job.setNames(names);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.setInput(input);
            job.setHorizontalFilp(horizontalFilp);
            job.setA(a);
            job.setB(b);
            job.setNoise(noise);
            job.reinitialize();
        }
        return job;
    }

    public static void load(String path, String extName, String[] names, int[] indexs, int batchSize, Tensor input, float[] a, float[] b, float[] noise, boolean normalization, boolean horizontalFilp) {
        //		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
        DiffusionImageLoader job = getInstance(path, extName, names, indexs, batchSize, input, a, b, noise, normalization, horizontalFilp, 0, batchSize - 1);
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
            DiffusionImageLoader left = new DiffusionImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), getA(), getB(), getNoise(), normalization, horizontalFilp, getStart(), mid - 1);
            DiffusionImageLoader right = new DiffusionImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), getA(), getB(), getNoise(), normalization, horizontalFilp, mid, getEnd());
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
            float[] data = YoloImageUtils.loadImgDataToArray(filePath, true, mean, std);
            if (horizontalFilp) {
                float[] newData = new float[data.length];
                data = DataTransforms.randomHorizontalFilp(data, newData, input.channel, input.height, input.width);
            }
            for (int j = 0; j < data.length; j++) {
                data[j] = data[j] * getA()[i] + getNoise()[i * data.length + j] * getB()[i];
            }
            System.arraycopy(data, 0, getInput().data, i * data.length, data.length);
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

    public float[] getA() {
        return a;
    }

    public void setA(float[] a) {
        this.a = a;
    }

    public float[] getB() {
        return b;
    }

    public void setB(float[] b) {
        this.b = b;
    }

    public float[] getNoise() {
        return noise;
    }

    public void setNoise(float[] noise) {
        this.noise = noise;
    }

    public boolean isHorizontalFilp() {
        return horizontalFilp;
    }

    public void setHorizontalFilp(boolean horizontalFilp) {
        this.horizontalFilp = horizontalFilp;
    }
}

