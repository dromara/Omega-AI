package com.omega.engine.parallel.dp;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.parallel.params.Parameters;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.dataset.SFTBinDataset;
import com.omega.example.transformer.dataset.parallel.ParallelDataLoader;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.runtime.JCuda;

public class DP {
	
	private int masterRank = 0;
	
	private int devices = 1;
	
	private int[] deviceIds;
	
	private NetworkType networkType;
	
	private Parameters parameters;
	
	private ParallelDataLoader pd;
	
	private int trainTime = 1;
	
	private Map<Integer,NetworkRunnable> threads = new HashMap<Integer, NetworkRunnable>();
	
	public DP(int[] deviceIds,NetworkType networkType,Parameters parameters,ParallelDataLoader pd,int trainTime) {
		this.pd = pd;
		this.deviceIds = deviceIds;
		this.devices = deviceIds.length;
		this.networkType = networkType;
		this.parameters = parameters;
		this.trainTime = trainTime;
	}
	
	public DP(int[] deviceIds,int masterRank,NetworkType networkType,Parameters parameters,ParallelDataLoader pd,int trainTime) {
		this.pd = pd;
		this.deviceIds = deviceIds;
		this.devices = deviceIds.length;
		this.masterRank = masterRank;
		this.networkType = networkType;
		this.parameters = parameters;
		this.trainTime = trainTime;
	}
	
	public void train() {
		
		int[] count = new int[1];
		
		JCuda.cudaGetDeviceCount(count);
		
		System.out.println("device count:"+count[0]);
		
		System.out.println("device:"+JsonUtils.toJson(deviceIds));
		
		ExecutorService executorService = Executors.newFixedThreadPool(devices);
		
		CyclicBarrier barrier = new CyclicBarrier(devices);
		
		for(int rankId:deviceIds) {
			boolean master = false;
			if(rankId == masterRank) {
				master = true;
			}
			NetworkRunnable thread = new NetworkRunnable(rankId, barrier, networkType, parameters, master, this);
			getThreads().put(rankId, thread);
			executorService.execute(thread);
		}
		
		executorService.shutdown();
		
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

	public Map<Integer,NetworkRunnable> getThreads() {
		return threads;
	}

	public void setThreads(Map<Integer,NetworkRunnable> threads) {
		this.threads = threads;
	}

	public ParallelDataLoader getPd() {
		return pd;
	}

	public int getTrainTime() {
		return trainTime;
	}
	
}
