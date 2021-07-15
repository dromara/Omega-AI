package com.omega.engine.updater;


/**
 * moving weighted average
 * 
 * @author Administrator
 *
 */
public class MWAUtils {
	
	public static double alpha = 0.9d;
	
	public static double[] mwa(double[] x,double[] runing){
		
		if(runing == null) {
			runing = new double[x.length];
		}
		
		for(int c = 0;c<runing.length;c++) {
			runing[c] = alpha * runing[c] + (1 - alpha) * x[c];
		}
		
		return runing;
	}
	
	public static double[][] mwa(double[][] x,double[][] runing){
		
		if(runing == null) {
			runing = new double[x.length][x[0].length];
		}
		
		for(int c = 0;c<runing.length;c++) {
			for(int h = 0;h<runing[c].length;h++) {
				runing[c][h] = alpha * runing[c][h] + (1 - alpha) * x[c][h];
			}
		}
		
		return runing;
	}
	
	public static double[][][] mwa(double[][][] x,double[][][] runing){
		
		if(runing == null) {
			runing = new double[x.length][x[0].length][x[0][0].length];
		}
		
		for(int c = 0;c<runing.length;c++) {
			for(int h = 0;h<runing[c].length;h++) {
				for(int w = 0;w<runing[c][h].length;w++) {
					runing[c][h][w] = alpha * runing[c][h][w] + (1 - alpha) * x[c][h][w];
				}
			}
		}
		
		return runing;
	}
	
	public static double[][][][] mwa(double[][][][] x,double[][][][] runing){
		
		if(runing == null) {
			runing = new double[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		}
		
		for(int n = 0;n<runing.length;n++) {
			for(int c = 0;c<runing[n].length;c++) {
				for(int h = 0;h<runing[n][c].length;h++) {
					for(int w = 0;w<runing[n][c][h].length;w++) {
						runing[n][c][h][w] = alpha * runing[n][c][h][w] + (1 - alpha) * x[n][c][h][w];
					}
				}
			}
		}
		
		return runing;
	}
	
	
}
