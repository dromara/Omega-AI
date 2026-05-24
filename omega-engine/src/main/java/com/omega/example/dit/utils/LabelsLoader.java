package com.omega.example.dit.utils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class LabelsLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static LabelsLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private BPETokenizerEN tokenizer;
    private List<Map<String, Object>> datas;
    private int[] indexs;
    private Tensor label;
    private int maxContextLen;

    public LabelsLoader(BPETokenizerEN tokenizer, List<Map<String, Object>> datas, int[] indexs, int batchSize, Tensor label, int maxContextLen, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setTokenizer(tokenizer);
        this.setDatas(datas);
        this.maxContextLen = maxContextLen;
        this.setIndexs(indexs);
        this.setLabel(label);
    }

    public static LabelsLoader getInstance(BPETokenizerEN tokenizer, List<Map<String, Object>> datas, int[] indexs, int batchSize, Tensor label, int maxContextLen, int start, int end) {
        if (job == null) {
            job = new LabelsLoader(tokenizer, datas, indexs, batchSize, label, maxContextLen, start, end);
        } else {
            if (label != job.getLabel()) {
                job.setLabel(label);
            }
            job.setTokenizer(tokenizer);
            job.setDatas(datas);
            job.setStart(0);
            job.setEnd(end);
            job.setIndexs(indexs);
            job.reinitialize();
        }
        return job;
    }

    public static void load(BPETokenizerEN tokenizer, List<Map<String, Object>> datas, int[] indexs, int batchSize, Tensor label, int maxContextLen) {
        //		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
        LabelsLoader job = getInstance(tokenizer, datas, indexs, batchSize, label, maxContextLen, 0, batchSize - 1);
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
            LabelsLoader left = null;
            LabelsLoader right = null;
            left = new LabelsLoader(tokenizer, datas, indexs, batchSize, label, maxContextLen, getStart(), mid - 1);
            right = new LabelsLoader(tokenizer, datas, indexs, batchSize, label, maxContextLen, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        	int idx = indexs[i];
            String text = datas.get(idx).get("label").toString();
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = tokenizer.eos();
                }
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

	public Tensor getLabel() {
		return label;
	}

	public void setLabel(Tensor label) {
		this.label = label;
	}

	public List<Map<String, Object>> getDatas() {
		return datas;
	}

	public void setDatas(List<Map<String, Object>> datas) {
		this.datas = datas;
	}

	public BPETokenizerEN getTokenizer() {
		return tokenizer;
	}

	public void setTokenizer(BPETokenizerEN tokenizer) {
		this.tokenizer = tokenizer;
	}
}

