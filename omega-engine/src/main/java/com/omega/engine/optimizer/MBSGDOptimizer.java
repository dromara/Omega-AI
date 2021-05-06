package com.omega.engine.optimizer;

import java.math.BigDecimal;
import java.util.concurrent.CountDownLatch;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
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
	
	public MBSGDOptimizer(Network network, int trainTime, double error,int batchSize) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = MatrixOperation.zero(this.network.outputNum);
		this.lossDiff = MatrixOperation.zero(this.network.outputNum);
	}

	public MBSGDOptimizer(Network network, int trainTime, double error,int batchSize,LearnRateUpdate learnRateUpdate) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = MatrixOperation.zero(this.network.outputNum);
		this.lossDiff = MatrixOperation.zero(this.network.outputNum);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	@Override
	public void train() {
		// TODO Auto-generated method stub

		try {
			
			for(int i = 0;i<this.trainTime;i++) {
				
				this.trainIndex = i;
				
				if(this.currentError <= this.error && this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.loss = MatrixOperation.clear(this.loss);
				
				this.lossDiff = MatrixOperation.clear(this.lossDiff);

				/**
				 * random data index
				 */
				int[] dataSetIndexs = MathUtils.randomInt(this.network.getTrainingData().dataSize - 1, this.batchSize);
				
				int successCount = 0;

				/**
				 * batch training
				 * using train engine
				 */
				if(this.getTrainEngine() != null) {
					
					CountDownLatch countDownLatch = new CountDownLatch(this.batchSize);
					
					for(int index:dataSetIndexs) {
						
						this.getTrainEngine().train(this, index, countDownLatch);
	
					}

					try {
						
			            countDownLatch.await();
			            
			        } catch (Exception e) {
			        	e.printStackTrace();
			            System.out.println("阻塞异常");
			        }
					
//			        System.out.println("执行完成.");
			        
				}else {
					
					for(int index:dataSetIndexs) {
						
						/**
						 * forward
						 */
						double[] output = this.network.forward(this.network.getTrainingData().dataInput[index]);
						
						/**
						 * loss
						 */
						this.loss = MatrixOperation.add(this.loss, this.network.loss(output, this.network.getTrainingData().dataLabel[index]));
						
						/**
						 * loss diff
						 */
						this.lossDiff = MatrixOperation.add(this.lossDiff, this.network.lossDiff(output, this.network.getTrainingData().dataLabel[index]));
						
						/**
						 * 计算正确率
						 */
						boolean isSame = LabelUtils.checkLabelForVector(output, this.network.getTrainingData().labelSet, this.network.getTrainingData().labels[index]);
						
						if(isSame) {
							successCount++;
						}
						
					}
					
				}
				
				this.loss = MatrixOperation.division(this.loss, this.network.getTrainingData().dataSize);
				
				this.lossDiff = MatrixOperation.division(this.lossDiff, this.network.getTrainingData().dataSize);
				
				/**
				 * current time error
				 */
				this.currentError = MatrixOperation.sum(this.loss);
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				/**
				 * back
				 */
				this.network.back(this.lossDiff);
				
				double accuracy = new BigDecimal(successCount).divide(new BigDecimal(dataSetIndexs.length)).doubleValue();
				
				System.out.println("training["+this.trainIndex+"] (lr:"+this.network.learnRate+") ["+(accuracy * 100) +"%"+"] currentError:"+this.currentError);
				
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
