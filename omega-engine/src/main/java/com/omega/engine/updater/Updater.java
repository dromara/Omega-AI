package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;

/**
 * Updater
 * 
 * @author Administrator
 *
 */
public abstract class Updater {
	
	public double beta = 0.9d;
	
	public double[][] vdw;
	
	public double[] vdb;
	
	public abstract void update(Layer layer);
	
}
