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
	
	public static float decay_rate = 0.7f;
	
	public static float lr = 0.1f;
	
	public static float decayedLR(float olr,int index) {
		olr = (float) (lr * 1.0d / (1.0d + decay_rate * (index + 1)));
		return olr;
	}
	
}
