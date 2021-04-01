package com.omega.engine.optimizer;

import com.omega.common.task.TaskEngine;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LRDecay;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * 
 * Optimizer
 * 
 * @author Administrator
 *
 */
public abstract class Optimizer {
	
	public int trainIndex = 0;
	
	public int batchSize = 0;
	
	public int dataSize = 0;
	
	public double[] loss;
	
	public double[] lossDiff;
	
	public int trainTime = 1000;
	
	public int minTrainTime = 1000;
	
	public double currentError = 1.0d;
	
	public double error = 0.01d;
	
	public Network network;
	
	private TaskEngine trainEngine;
	
	public LearnRateUpdate learnRateUpdate = LearnRateUpdate.NONE;
	
	public abstract void train();
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 */
	public Optimizer(Network network,int trainTime,double error) {
		this.network = network;
		this.trainTime = trainTime;
		this.error = error;
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 */
	public Optimizer(Network network,int trainTime,int minTrainTime,double error) {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 */
	public Optimizer(Network network,int trainTime,int minTrainTime,double error,LearnRateUpdate learnRateUpdate) {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.learnRateUpdate = learnRateUpdate;
	}
	
	public void updateLR() {
		switch (this.learnRateUpdate) {
		case LR_DECAY:
			this.network.learnRate = LRDecay.decayedLR(this.network.learnRate, this.trainIndex);
			break;
		case NONE:
			break;
		}
	}
	
	public void setTrainEngine(TaskEngine trainEngine) {
		this.trainEngine = trainEngine;
	}
	
	public TaskEngine getTrainEngine() {
		return this.trainEngine;
	}
	
}
