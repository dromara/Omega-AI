package com.omega.engine.optimizer.lr;


/**
 * GDDecay
 * 
 * lr_i = lr_start * 1.0 / (1.0 + decay * i)
 * decay => [0.0,1.0]
 * @author Administrator
 *
 */
public class GDDecay {
	
	public static double decay_rate = 0.7d;
	
	public static double lr = 0.1d;
	
	public static double decayedLR(double olr,int index) {
		olr = lr * 1.0d / (1.0d + decay_rate * (index + 1));
		return olr;
	}
	
}
