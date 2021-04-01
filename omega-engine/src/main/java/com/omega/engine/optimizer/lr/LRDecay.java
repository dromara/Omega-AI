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
	
	public static double decay_rate = 0.99d;
	
	public static double lr = 0.1d;
	
	public static double decayedLR(double olr,int index) {
		if(index % decay_steps == 0) {
			olr = lr * Math.pow(decay_rate, (index/decay_steps));
		}
		return olr;
	}
	
}
