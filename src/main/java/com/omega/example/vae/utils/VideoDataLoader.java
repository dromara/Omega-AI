package com.omega.example.vae.utils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.YoloImageUtils;

/**
 * VideoDataLoader
 *
 * @author Administrator
 */
public class VideoDataLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static VideoDataLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private String path;
    private List<Map<String, String>> datas;
    private int[] indexs;
    private int depth;
    private Tensor input;
    private String extName;
    private boolean normalization = false;
    private float[] mean;
    private float[] std;

    public VideoDataLoader(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.depth = depth;
        this.setPath(path);
        this.extName = extName;
        this.setDatas(datas);
        this.setIndexs(indexs);
        this.setInput(input);
        this.normalization = normalization;
    }

    public VideoDataLoader(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization, float[] mean, float[] std, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.depth = depth;
        this.setPath(path);
        this.extName = extName;
        this.setDatas(datas);
        this.setIndexs(indexs);
        this.setInput(input);
        this.mean = mean;
        this.std = std;
        this.normalization = normalization;
    }

    public static VideoDataLoader getInstance(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization, int start, int end) {
        if (job == null) {
            job = new VideoDataLoader(path, extName, datas, indexs, batchSize, depth, input, normalization, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
            }
            job.setPath(path);
            job.setDatas(datas);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.reinitialize();
            
        }
        return job;
    }

    public static VideoDataLoader getInstance(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization, float[] mean, float[] std, int start, int end) {
        if (job == null) {
            job = new VideoDataLoader(path, extName, datas, indexs, batchSize, depth, input, normalization, mean, std, start, end);
        } else {
            if (input != job.getInput()) {
                job.setInput(input);
            }
            job.setPath(path);
            job.setDatas(datas);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.reinitialize();
        }
        return job;
    }

    public static void load(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization) {
        //		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
        VideoDataLoader job = getInstance(path, extName, datas, indexs, batchSize, depth, input, normalization, 0, batchSize - 1);
        ForkJobEngine.run(job);
    }

    public static void load(String path, String extName, List<Map<String, String>> datas, int[] indexs, int batchSize, int depth, Tensor input, boolean normalization, float[] mean, float[] std) {
        VideoDataLoader job = getInstance(path, extName, datas, indexs, batchSize, depth, input, normalization, mean, std, 0, batchSize - 1);
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
            VideoDataLoader left = null;
            VideoDataLoader right = null;
            if (mean != null) {
                left = new VideoDataLoader(getPath(), extName, datas, getIndexs(), batchSize, depth, getInput(), normalization, mean, std, getStart(), mid - 1);
                right = new VideoDataLoader(getPath(), extName, datas, getIndexs(), batchSize, depth, getInput(), normalization, mean, std, mid, getEnd());
            } else {
                left = new VideoDataLoader(getPath(), extName, datas, getIndexs(), batchSize, depth, getInput(), normalization, getStart(), mid - 1);
                right = new VideoDataLoader(getPath(), extName, datas, getIndexs(), batchSize, depth, getInput(), normalization, mid, getEnd());
            }
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        	
        	int index = getIndexs()[i];
        	
        	Map<String, String> once = getDatas().get(index);
        	
        	String filename = once.get("filename");
        	
        	int num_frames = Integer.parseInt(once.get("num_frames"));
        	
        	int startDepth = RandomUtils.randomInt(0, num_frames - depth - 2);
        	
        	int onceLen = depth * 3 * getInput().height * getInput().width;
        	
        	for(int idx = 0;idx<depth;idx++) {
    			String filePath = getPath() + "/" + filename + "/" + (startDepth + idx) + extName;
    			float[] data = null;
                if (mean != null) {
                    data = YoloImageUtils.loadImgDataToArray(filePath, normalization, mean, std);
                } else {
                    data = YoloImageUtils.loadImgDataToArray(filePath, normalization);
                }
                if(data == null) {
                	throw new RuntimeException(filePath + " is not extis.");
                }
                int start = i * onceLen + idx * 3 * getInput().height * getInput().width;
                System.arraycopy(data, 0, getInput().data, start, data.length);
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

    public Tensor getInput() {
        return input;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

	public List<Map<String, String>> getDatas() {
		return datas;
	}

	public void setDatas(List<Map<String, String>> datas) {
		this.datas = datas;
	}
}

