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
	
	public float beta = 0.9f;
	
	public float lr = 0.0001f;
	
	public float[][] vdw;
	
	public float[] vdgama;
	
	public float[] vdb;
	
	public float[][][][] vdmw;  //c * kn * kh * kw
	
	public float[] vdmb;
	
	public abstract void update(Layer layer);
	
	public abstract void updateForMatrix(Layer layer);
	
	public abstract void updateForBN(BNLayer layer);
	
	public abstract UpdaterType getUpdaterType();
	
}
