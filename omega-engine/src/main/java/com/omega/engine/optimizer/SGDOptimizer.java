package com.omega.engine.optimizer;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * Stochastic Gradient Descent
 * 
 * @author Administrator
 *
 */
public class SGDOptimizer extends Optimizer {
	
	public SGDOptimizer(Network network, int trainTime, double error) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
	}
	
	public SGDOptimizer(Network network, int trainTime, double error,LearnRateUpdate learnRateUpdate) {
		super(network, trainTime, error);
		// TODO Auto-generated constructor stub
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
				
				/**
				 * random data index
				 */
				int dataIndex = MathUtils.randomInt(this.network.getTrainingData().dataSize - 1);
				
				/**
				 * forward
				 */
				double[] output = this.network.forward(this.network.getTrainingData().dataInput[dataIndex]);

				/**
				 * loss
				 */
				this.loss = this.network.loss(output, this.network.getTrainingData().dataLabel[dataIndex]);
				
				/**
				 * current time error
				 */
				this.currentError = MatrixOperation.sum(this.loss);

				/**
				 * loss diff
				 */
				this.lossDiff = this.network.lossDiff(output, this.network.getTrainingData().dataLabel[dataIndex]);

				/**
				 * back
				 */
				this.network.back(this.lossDiff);
				
				this.trainIndex = i;
				
				System.out.println("training["+this.trainIndex+"] currentError:"+this.currentError);
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
