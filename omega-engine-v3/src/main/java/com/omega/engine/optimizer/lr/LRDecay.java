package com.omega.engine.optimizer.lr;


/**
 * 
 * LRDecay
 * 
 * @author Administrator
 *	
 * @remak
 * 
 * decayed_learning_rate = learning_rate * decay_rate ^ (global_steps/decay_steps)
 *
 */
public class LRDecay {
	
	public static int decay_steps = 100;
	
	public static float decay_rate = 0.99f;
	
	public static float lr = 0.1f;
	
	public static float decayedLR(float olr,int index) {
		if(index % decay_steps == 0) {
			olr = (float) (lr * Math.pow(decay_rate, (index/decay_steps)));
		}
		return olr;
	}
	
}
