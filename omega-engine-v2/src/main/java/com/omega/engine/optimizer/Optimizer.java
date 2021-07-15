package com.omega.engine.optimizer;

import com.omega.common.task.TaskEngine;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.GDDecay;
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
	
	public int batchSize = 1;
	
	public int dataSize = 0;
	
	public Blob loss;
	
	public Blob lossDiff;
	
	public int trainTime = 1000;
	
	public int minTrainTime = 1000;
	
	public double currentError = 1.0d;
	
	public double error = 0.01d;
	
	public Network network;
	
	private TaskEngine trainEngine;
	
	public LearnRateUpdate learnRateUpdate = LearnRateUpdate.NONE;
	
	private BaseData trainingData;
	
	private BaseData testData;
	
	public abstract void train(BaseData trainingData);
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,double error) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.error = error;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,double error) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,double error,LearnRateUpdate learnRateUpdate) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.learnRateUpdate = learnRateUpdate;
		this.network.init();
	}
	
	public void setTrainEngine(TaskEngine trainEngine) {
		this.trainEngine = trainEngine;
	}
	
	public TaskEngine getTrainEngine() {
		return this.trainEngine;
	}

	public BaseData getTrainingData() {
		return trainingData;
	}

	public void setTrainingData(BaseData trainingData) {
		this.trainingData = trainingData;
	}

	public BaseData getTestData() {
		return testData;
	}

	public void setTestData(BaseData testData) {
		this.testData = testData;
	}
	
	public void updateLR() {
		switch (this.learnRateUpdate) {
		case LR_DECAY:
			this.network.learnRate = LRDecay.decayedLR(this.network.learnRate, this.trainIndex);
			break;
		case GD_GECAY:
			this.network.learnRate = GDDecay.decayedLR(this.network.learnRate, this.trainIndex);
			break;
		case NONE:
			break;
		}
	}
	
	public double test(BaseData testData) {
		// TODO Auto-generated method stub
		double error = 0.0d;
		
		double trueCount = 0;

		for(int n = 0;n<testData.number;n++) {
			
//			double[] onceError = MatrixOperation.subtraction(output, testData.dataLabel[i]);

			Blob output = this.network.predict(testData.getOnceData(n));

			String label = testData.labels[n];
			
			String predictLabel = LabelUtils.vectorTolabel(output.maxtir[0][0][0], testData.labelSet);
			
			if(!label.equals(predictLabel)) {
//				System.out.println("index:"+n+"::"+JsonUtils.toJson(output)+"==>predictLabel:"+predictLabel+"==label:"+label+":"+label.equals(predictLabel));
			}else {
				trueCount++;
			}
			
		}
		
		error = trueCount / testData.number;
		
		System.out.println("准确率:"+ error * 100 +"%");
		
		return error;
	}
	
	public double accuracy(Blob output,double[][] labels,String[] labelSet) {
		
		double error = 0.0d;
		double trueCount = 0;
		
		for(int n = 0;n<output.number;n++) {

			String label = LabelUtils.vectorTolabel(labels[n], labelSet);
			
			String predictLabel = LabelUtils.vectorTolabel(output.maxtir[n][0][0], labelSet);
			
			if(label.equals(predictLabel)) {
				trueCount++;
			}
			
		}
		
		error = trueCount / output.number * 100;
//		
//		System.out.println("准确率:"+ error * 100 +"%");
//		
		return error;
	}
	
}
