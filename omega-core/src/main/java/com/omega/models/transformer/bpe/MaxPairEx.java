package com.omega.models.transformer.bpe;

import com.omega.common.task.ForkJobEngine;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * FileDataLoader
 *
 * @author Administrator
 */
public class MaxPairEx extends RecursiveAction {
    /**
     *
     */
    private static final long serialVersionUID = 6302699701667951010L;
    private static MaxPairEx job;
    private int start = 0;
    private int end = 0;
    private List<Integer> ids;
    private Map<String, Integer> pairCountMap;

    public MaxPairEx(List<Integer> ids, Map<String, Integer> pairCountMap, int start, int end) {
        this.setStart(start);
        this.setEnd(end);
        this.ids = ids;
        this.pairCountMap = pairCountMap;
    }

    public static MaxPairEx getInstance(List<Integer> ids, Map<String, Integer> pairCountMap, int start, int end) {
        if (job == null) {
            job = new MaxPairEx(ids, pairCountMap, start, end);
        } else {
            if (pairCountMap != job.getPairCountMap()) {
                job.setPairCountMap(pairCountMap);
            }
            if (ids != job.getIds()) {
                job.setIds(ids);
            }
            job.setStart(0);
            job.setEnd(end);
            job.reinitialize();
        }
        return job;
    }

    public static void load(List<Integer> ids, Map<String, Integer> pairCountMap) {
        MaxPairEx job = getInstance(ids, pairCountMap, 0, ids.size() - 2);
        ForkJobEngine.run(job);
    }

    public static String getMaxKey(List<Integer> ids, Map<String, Integer> pairCountMap) {
        pairCountMap.clear();
        load(ids, pairCountMap);
        return pairCountMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    @Override
    protected void compute() {
        // TODO Auto-generated method stub
        int length = getEnd() - getStart() + 1;
        if (length < 8 || length <= ids.size() / 8) {
            load();
        } else {
            int mid = (getStart() + getEnd() + 1) >>> 1;
            MaxPairEx left = new MaxPairEx(ids, pairCountMap, getStart(), mid - 1);
            MaxPairEx right = new MaxPairEx(ids, pairCountMap, mid, getEnd());
            ForkJoinTask<Void> leftTask = left.fork();
            ForkJoinTask<Void> rightTask = right.fork();
            leftTask.join();
            rightTask.join();
        }
    }

    private void load() {
        for (int i = getStart(); i <= getEnd(); i++) {
            String pairKey = ids.get(i) + ":" + ids.get(i + 1);
            pairCountMap.put(pairKey, pairCountMap.getOrDefault(pairKey, 0) + 1);
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

    public Map<String, Integer> getPairCountMap() {
        return pairCountMap;
    }

    public void setPairCountMap(Map<String, Integer> pairCountMap) {
        this.pairCountMap = pairCountMap;
    }

    public List<Integer> getIds() {
        return ids;
    }

    public void setIds(List<Integer> ids) {
        this.ids = ids;
    }
}

