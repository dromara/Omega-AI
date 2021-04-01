package com.omega.common.task;

import java.util.concurrent.CountDownLatch;

import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.optimizer.Optimizer;

@Component
public class TaskEngine {
	
	@Async("taskAsync")
	public void train(Optimizer op,int index,CountDownLatch countDownLatch) {
		
		try {
			
			synchronized (op.network) {
				
				/**
				 * forward
				 */
				double[] output = op.network.forward(op.network.getTrainingData().dataInput[index]);

				/**
				 * loss
				 */
				op.loss = MatrixOperation.add(op.loss, op.network.loss(output, op.network.getTrainingData().dataLabel[index]));
				
				/**
				 * loss diff
				 */
				op.lossDiff = MatrixOperation.add(op.lossDiff, op.network.lossDiff(output, op.network.getTrainingData().dataLabel[index]));
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {
			
			countDownLatch.countDown();
			
		}
//		
//		return new AsyncResult<Boolean>(true);
	}
	
}
