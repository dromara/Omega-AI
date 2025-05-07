package com.omega.example.transformer.utils.bpe;

import com.omega.common.task.ForkJobEngine;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class MergeEx2 extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static MergeEx2 job;
    private int start = 0;
    private int end = 0;
    private CopyOnWriteArrayList<Integer> ids;
    private String pair;
    private int idx;

    public MergeEx2(CopyOnWriteArrayList<Integer> ids, String pair, int idx, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.ids = ids;
        this.pair = pair;
        this.idx = idx;
    }

    public static MergeEx2 getInstance(CopyOnWriteArrayList<Integer> ids, String pair, int idx, int start, int end) {
        if (job == null) {
            job = new MergeEx2(ids, pair, idx, start, end);
        } else {
            if (ids != job.getIds()) {
                job.setIds(ids);
            }
            job.setPair(pair);
            job.setIdx(idx);
            job.setStart(0);
            job.setEnd(end);
            job.reinitialize();
        }
        return job;
    }

    public static void load(CopyOnWriteArrayList<Integer> ids, String pair, int idx) {
        MergeEx2 job = getInstance(ids, pair, idx, 0, ids.size() - 2);
        ForkJobEngine.run(job);
    }

    public static CopyOnWriteArrayList<Integer> merge(CopyOnWriteArrayList<Integer> ids, String pair, int idx) {
        load(ids, pair, idx);
        return ids;
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= ids.size() / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            MergeEx2 left = new MergeEx2(ids, pair, idx, getStart(), mid - 1);
            MergeEx2 right = new MergeEx2(ids, pair, idx, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String pairKey = ids.get(i) + ":" + ids.get(i + 1);
            if (pairKey.equals(pair)) {
                ids.set(i, idx);
                ids.remove(i + 1);
                i++;
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

    public CopyOnWriteArrayList<Integer> getIds() {
        return ids;
    }

    public void setIds(CopyOnWriteArrayList<Integer> ids) {
        this.ids = ids;
    }

    public String getPair() {
        return pair;
    }

    public void setPair(String pair) {
        this.pair = pair;
    }

    public int getIdx() {
        return idx;
    }

    public void setIdx(int idx) {
        this.idx = idx;
    }
}

