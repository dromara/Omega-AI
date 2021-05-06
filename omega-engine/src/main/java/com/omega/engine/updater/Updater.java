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
	
	public double[][][][] vdmw;  //c * kn * kh * kw
	
	public double[] vdmb;
	
	public abstract void update(Layer layer);
	
	public abstract void updateForMatrix(Layer layer);
	
}
