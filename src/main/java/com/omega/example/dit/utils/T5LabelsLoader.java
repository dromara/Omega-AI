package com.omega.example.dit.utils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.utils.SentencePieceTokenizer;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class T5LabelsLoader extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static T5LabelsLoader job;
    private int start = 0;
    private int end = 0;
    private int batchSize = 0;
    private SentencePieceTokenizer tokenizer;
    private List<Map<String, Object>> datas;
    private int[] indexs;
    private Tensor label;
    private Tensor mask;
   
	private int maxContextLen;
    private String key;

    public T5LabelsLoader(SentencePieceTokenizer tokenizer, List<Map<String, Object>> datas, String key, int[] indexs, int batchSize, Tensor label, Tensor mask, int maxContextLen, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.batchSize = batchSize;
        this.setTokenizer(tokenizer);
        this.setDatas(datas);
        this.maxContextLen = maxContextLen;
        this.setIndexs(indexs);
        this.setLabel(label);
        this.setMask(mask);
        this.key = key;
    }

    public static T5LabelsLoader getInstance(SentencePieceTokenizer tokenizer, List<Map<String, Object>> datas, String key, int[] indexs, int batchSize, Tensor label, Tensor mask, int maxContextLen, int start, int end) {
        if (job == null) {
            job = new T5LabelsLoader(tokenizer, datas, key, indexs, batchSize, label, mask, maxContextLen, start, end);
        } else {
            if (label != job.getLabel()) {
                job.setLabel(label);
                job.setMask(mask);
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

    public static void load(SentencePieceTokenizer tokenizer, List<Map<String, Object>> datas, String key, int[] indexs, int batchSize, Tensor label, Tensor mask, int maxContextLen) {
        T5LabelsLoader job = getInstance(tokenizer, datas, key, indexs, batchSize, label, mask, maxContextLen, 0, batchSize - 1);
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
            T5LabelsLoader left = null;
            T5LabelsLoader right = null;
            left = new T5LabelsLoader(tokenizer, datas, key, indexs, batchSize, label, mask, maxContextLen, getStart(), mid - 1);
            right = new T5LabelsLoader(tokenizer, datas, key, indexs, batchSize, label, mask, maxContextLen, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        	int idx = indexs[i];
            String text = datas.get(idx).get(key).toString();
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            for (int j = 0; j < maxContextLen; j++) {
            	int val = ids[j];
            	label.data[i * maxContextLen + j] = val;
            	if(val != tokenizer.pad()) {
            		mask.data[i * maxContextLen + j] = 0;
            	}else {
            		mask.data[i * maxContextLen + j] = -3.4028e+38f;
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

	public SentencePieceTokenizer getTokenizer() {
		return tokenizer;
	}

	public void setTokenizer(SentencePieceTokenizer tokenizer) {
		this.tokenizer = tokenizer;
	}
	
	public Tensor getMask() {
		return mask;
	}

	public void setMask(Tensor mask) {
		this.mask = mask;
	}

}

