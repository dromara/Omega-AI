package com.omega.models.transformer.bpe;

import com.omega.common.task.ForkJobEngine;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class MergeEx extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static MergeEx job;
    private int start = 0;
    private int end = 0;
    private List<Integer> ids;
    private String pair;
    private int idx;

    public MergeEx(List<Integer> ids, String pair, int idx, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.ids = ids;
        this.pair = pair;
        this.idx = idx;
    }

    public static MergeEx getInstance(List<Integer> ids, String pair, int idx, int start, int end) {
        if (job == null) {
            job = new MergeEx(ids, pair, idx, start, end);
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

    public static void load(List<Integer> ids, String pair, int idx) {
        MergeEx job = getInstance(ids, pair, idx, 0, ids.size() - 2);
        ForkJobEngine.run(job);
    }

    public static List<Integer> merge(List<Integer> ids, String pair, int idx) {
        load(ids, pair, idx);
        //		long start3 = System.nanoTime();
        ids.removeIf(Objects::isNull);
        //		System.out.println("remove:"+(System.nanoTime()-start3)/1e6+"ms.");
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
            MergeEx left = new MergeEx(ids, pair, idx, getStart(), mid - 1);
            MergeEx right = new MergeEx(ids, pair, idx, mid, getEnd());
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
                ids.set(i + 1, null);
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

    public List<Integer> getIds() {
        return ids;
    }

    public void setIds(List<Integer> ids) {
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

