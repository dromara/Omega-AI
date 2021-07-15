package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.BNLayer;

/**
 * Updater
 * 
 * @author Administrator
 *
 */
public abstract class Updater {
	
	public UpdaterType updaterType;
	
	public double beta = 0.9d;
	
	public double[][] vdw;
	
	public double[] vdgama;
	
	public double[] vdb;
	
	public double[][][][] vdmw;  //c * kn * kh * kw
	
	public double[] vdmb;
	
	public abstract void update(Layer layer);
	
	public abstract void updateForMatrix(Layer layer);
	
	public abstract void updateForBN(BNLayer layer);
	
	public abstract UpdaterType getUpdaterType();
	
}
