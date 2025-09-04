package com.omega.example.transformer.utils.tokenizers;

import com.omega.common.task.ForkJobEngine;

import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class EncodeExMaxLen extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static EncodeExMaxLen job;
    private int start = 0;
    private int end = 0;
    private int maxLen;
    private boolean keep = true;
    private List<String> txtList;
    private Tokenizer tokenizer;
    private String[] idxList;

    public EncodeExMaxLen(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.setIdxList(idxList);
        this.setMaxLen(maxLen);
        this.txtList = txtList;
        this.tokenizer = tokenizer;
    }
    
    public EncodeExMaxLen(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen, boolean keep, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.setIdxList(idxList);
        this.setMaxLen(maxLen);
        this.setKeep(keep);
        this.txtList = txtList;
        this.tokenizer = tokenizer;
    }

    public static EncodeExMaxLen getInstance(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen, int start, int end) {
        if (job == null) {
            job = new EncodeExMaxLen(txtList, idxList, tokenizer, maxLen, start, end);
        } else {
            if (txtList != job.getTxtList()) {
                job.setTxtList(txtList);
            }
            job.setIdxList(idxList);
            job.setTokenizer(tokenizer);
            job.setMaxLen(maxLen);
            job.setStart(0);
            job.setEnd(end);
            job.reinitialize();
        }
        return job;
    }
    
    public static EncodeExMaxLen getInstance(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen, boolean keep, int start, int end) {
        if (job == null) {
            job = new EncodeExMaxLen(txtList, idxList, tokenizer, maxLen, keep, start, end);
        } else {
            if (txtList != job.getTxtList()) {
                job.setTxtList(txtList);
            }
            job.setIdxList(idxList);
            job.setTokenizer(tokenizer);
            job.setMaxLen(maxLen);
            job.setStart(0);
            job.setEnd(end);
            job.setKeep(keep);
            job.reinitialize();
        }
        return job;
    }

    public static void encode(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen) {
        //		System.out.println("encoding.");
        EncodeExMaxLen job = getInstance(txtList, idxList, tokenizer, maxLen, 0, txtList.size() - 1);
        ForkJobEngine.run(job);
        //		System.out.println("encode finish.");
    }
    
    public static void encode(List<String> txtList, String[] idxList, Tokenizer tokenizer, int maxLen, boolean keep) {
        //		System.out.println("encoding.");
        EncodeExMaxLen job = getInstance(txtList, idxList, tokenizer, maxLen, keep, 0, txtList.size() - 1);
        ForkJobEngine.run(job);
        //		System.out.println("encode finish.");
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= txtList.size() / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            EncodeExMaxLen left = new EncodeExMaxLen(txtList, idxList, tokenizer, maxLen, getStart(), mid - 1);
            EncodeExMaxLen right = new EncodeExMaxLen(txtList, idxList, tokenizer, maxLen, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
        	int[] ids;
        	if(keep) {
        		ids = tokenizer.encodeInt(txtList.get(i), maxLen);
        	}else {
        		ids = tokenizer.encodeInt(txtList.get(i));
        	}
            String txt = "";
            for (int id : ids) {
                txt += id + " ";
            }
            if(keep) {
            	idxList[i] = txt;
            }else if(ids.length > maxLen){
            	idxList[i] = "-1";
            }else {
            	idxList[i] = txt;
            }
            //			System.out.println("encode["+i+"]finish.");
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

    public List<String> getTxtList() {
        return txtList;
    }

    public void setTxtList(List<String> txtList) {
        this.txtList = txtList;
    }

    public void setIdxList(String[] idxList) {
        this.idxList = idxList;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public void setTokenizer(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public int getMaxLen() {
        return maxLen;
    }

    public void setMaxLen(int maxLen) {
        this.maxLen = maxLen;
    }

	public boolean isKeep() {
		return keep;
	}

	public void setKeep(boolean keep) {
		this.keep = keep;
	}
}

