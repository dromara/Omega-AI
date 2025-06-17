package com.omega.engine.parallel.dp;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.params.Parameters;
import com.omega.example.transformer.dataset.parallel.ParallelDataLoader;
import jcuda.runtime.JCuda;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DP {
    private int masterRank = 0;
    private int devices = 1;
    private int[] deviceIds;
    private NetworkType networkType;
    private Parameters parameters;
    private ParallelDataLoader pd;
    private int trainTime = 1;
    private Map<Integer, NetworkRunnable> threads = new HashMap<Integer, NetworkRunnable>();
    private String pretrainModelPath;

    public DP(int[] deviceIds, NetworkType networkType, Parameters parameters, ParallelDataLoader pd, int trainTime) {
        this.pd = pd;
        this.deviceIds = deviceIds;
        this.devices = deviceIds.length;
        this.networkType = networkType;
        this.parameters = parameters;
        this.trainTime = trainTime;
    }

    public DP(int[] deviceIds, int masterRank, NetworkType networkType, Parameters parameters, ParallelDataLoader pd, int trainTime) {
        this.pd = pd;
        this.deviceIds = deviceIds;
        this.devices = deviceIds.length;
        this.masterRank = masterRank;
        this.networkType = networkType;
        this.parameters = parameters;
        this.trainTime = trainTime;
    }

    public NetworkRunnable getRight(int rankId) {
        if (rankId < deviceIds.length - 1) {
            return threads.get(rankId + 1);
        } else {
            return threads.get(0);
        }
    }

    public NetworkRunnable getLeft(int rankId) {
        if (rankId > 0) {
            return threads.get(rankId - 1);
        } else {
            return threads.get(deviceIds[deviceIds.length - 1]);
        }
    }

    public void load(String pretrainModelPath) {
        this.pretrainModelPath = pretrainModelPath;
    }

    public void train() {
        int[] count = new int[1];
        JCuda.cudaGetDeviceCount(count);
        System.out.println("device count:" + count[0]);
        System.out.println("device:" + JsonUtils.toJson(deviceIds));
        ExecutorService executorService = Executors.newFixedThreadPool(devices);
        CyclicBarrier barrier = new CyclicBarrier(devices);
        List<NetworkRunnable> threads = new ArrayList<NetworkRunnable>();
        for (int rankId : deviceIds) {
            boolean master = false;
            if (rankId == masterRank) {
                master = true;
            }
            NetworkRunnable thread = new NetworkRunnable(rankId, barrier, networkType, parameters, master, this);
            getThreads().put(rankId, thread);
            threads.add(thread);
        }
        try {
            executorService.invokeAll(threads);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            executorService.shutdown();
        }
    }

    public NetworkType getNetworkType() {
        return networkType;
    }

    public void setNetworkType(NetworkType networkType) {
        this.networkType = networkType;
    }

    public Parameters getParameters() {
        return parameters;
    }

    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public int getDevices() {
        return devices;
    }

    public void setDevices(int devices) {
        this.devices = devices;
    }

    public Map<Integer, NetworkRunnable> getThreads() {
        return threads;
    }

    public void setThreads(Map<Integer, NetworkRunnable> threads) {
        this.threads = threads;
    }

    public ParallelDataLoader getPd() {
        return pd;
    }

    public int getTrainTime() {
        return trainTime;
    }

    public Network getMaster() {
        return threads.get(masterRank).getNetwork();
    }

    public String getPretrainModelPath() {
        return pretrainModelPath;
    }
}

