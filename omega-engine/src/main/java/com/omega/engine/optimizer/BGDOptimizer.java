package com.omega.engine.optimizer;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * Batch Gradient Descent
 * 
 * @author Administrator
 *
 */
public class BGDOptimizer extends Optimizer {

	public BGDOptimizer(Network network, int trainTime, double error) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
		this.loss = MatrixOperation.zero(this.network.outputNum);
		this.lossDiff = MatrixOperation.zero(this.network.outputNum);
	}
	
	public BGDOptimizer(Network network, int trainTime, double error,LearnRateUpdate learnRateUpdate) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
		this.loss = MatrixOperation.zero(this.network.outputNum);
		this.lossDiff = MatrixOperation.zero(this.network.outputNum);
		this.learnRateUpdate = learnRateUpdate;
	}

	@Override
	public void train() {
		// TODO Auto-generated method stub
		try {
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.currentError <= this.error && this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.loss = MatrixOperation.clear(this.loss);
				
				this.lossDiff = MatrixOperation.clear(this.lossDiff);
				
				/**
				 * batch training
				 */
				for(int index = 0;index<this.network.getTrainingData().dataSize;index++) {
					
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
					
				}
				
				this.loss = MatrixOperation.division(this.loss, this.network.getTrainingData().dataSize);
				
				this.lossDiff = MatrixOperation.division(this.lossDiff, this.network.getTrainingData().dataSize);
				
				/**
				 * current time error
				 */
				this.currentError = MatrixOperation.sum(this.loss);

				/**
				 * back
				 */
				this.network.back(this.lossDiff);
				
				this.trainIndex = i;
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

}
