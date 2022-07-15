package com.omega.engine.optimizer;

import com.omega.common.task.TaskEngine;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.RandomUtils;
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
	
	private String sid;
	
	public int batchIndex = 1;
	
	public int trainIndex = 1;
	
	public int batchSize = 1;
	
	public int dataSize = 0;
	
	public Blob loss;
	
	public Blob lossDiff;
	
	public int trainTime = 10;
	
	public int minTrainTime = 100;
	
	public float currentError = 1.0f;
	
	public float error = 0.01f;
	
	public Network network;
	
	private TaskEngine trainEngine;
	
	public LearnRateUpdate learnRateUpdate = LearnRateUpdate.NONE;
	
	private BaseData trainingData;
	
	private BaseData testData;
	
	private boolean warmUp = false;
	
	public int burnIn = 300;
	
	public int power = 4;
	
	public float scale = 0.1f;
	
	public int step = 500;
	
	public float gama = 0.9999f;
	
	public float lr = 0.1f;
	
	public boolean isOnline = false;
	
	public abstract void train(BaseData trainingData);
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,float error,boolean warmUp) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.error = error;
		this.lr = network.learnRate;
		this.warmUp = warmUp;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,float error,boolean warmUp) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.warmUp = warmUp;
		this.lr = network.learnRate;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,float error,boolean warmUp,LearnRateUpdate learnRateUpdate) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.warmUp = warmUp;
		this.learnRateUpdate = learnRateUpdate;
		this.lr = network.learnRate;
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
		
		if(warmUp && batchIndex < burnIn) {
			this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
		}else {

			switch (this.learnRateUpdate) {
			case LR_DECAY:
				this.network.learnRate = LRDecay.decayedLR(this.network.learnRate, this.trainIndex);
				break;
			case GD_GECAY:
				this.network.learnRate = GDDecay.decayedLR(this.network.learnRate, this.trainIndex);
				break;
			case NONE:
				break;
			case CONSTANT:
				break;
			case RANDOM:
				this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
				break;
			case POLY:
				float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
				
				this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
				break;
			case STEP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
				break;
			case EXP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
				break;
			case SIG:
				this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
				break;
			}
			
		}
		
	}
	
	public float test(BaseData testData) {
		// TODO Auto-generated method stub
		float error = 0.0f;
		
		float trueCount = 0;

		for(int n = 0;n<testData.number;n++) {
			
//			float[] onceError = MatrixOperation.subtraction(output, testData.dataLabel[i]);

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
	
	public float accuracy(Blob output,float[][] labels,String[] labelSet) {
		
		float error = 0.0f;
		float trueCount = 0;
		
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

	public boolean isWarmUp() {
		return warmUp;
	}

	public void setWarmUp(boolean warmUp) {
		this.warmUp = warmUp;
	}
	
	public void online(boolean isOnline) {
		this.isOnline = isOnline;
	}

	public String getSid() {
		return sid;
	}

	public void setSid(String sid) {
		this.sid = sid;
	}
	
}
