package com.omega.engine.optimizer;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.controller.TrainTask;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * 
 * Mini Batch Stochastic Gradient Descent
 * 
 * @author Administrator
 *
 */
public class MBSGDOptimizer extends Optimizer {
	
	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}

	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = Blobs.blob(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	@Override
	public void train(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.currentTimeMillis();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					Blob input = trainingData.getRandomData(indexs[it]); 
					
					/**
					 * forward
					 */
					Blob output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, input.labels);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, input.labels);
					
					/**
					 * current time error
					 */
					this.currentError = MatrixOperation.sum(this.loss.maxtir) / this.batchSize;
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					float error = this.accuracy(output, input.labels, trainingData.labelSet);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.currentTimeMillis() - start)+"ms.]";
					
					System.out.println(msg);
					
					/**
					 * 发送消息
					 */
					if(isOnline && this.getSid() != null) {
						
						TrainTask.sendMsg(this.getSid(), msg);
						
					}
					
//					/**
//					 * update learning rate
//					 */
					this.updateLR();
					
					this.batchIndex++;
				}
//
//				/**
//				 * update learning rate
//				 */
//				this.updateLR();
//				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

}
