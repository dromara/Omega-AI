package com.omega.engine.pooling;

/**
 * 
 * @author Administrator
 *
 */
public abstract class PoolingFunction {
	
	public PoolingType poolingType;
	
	public double[][][][] input;
	
	public double[][][][] output;
	
	public double[][][] diff;
	
	public abstract double[][][][] active(double[][][][] x);
	
	public abstract double[][][] diff();
	
	public PoolingFunction(){}
	
}
