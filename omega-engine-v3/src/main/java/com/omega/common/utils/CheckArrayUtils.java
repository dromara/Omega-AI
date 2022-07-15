package com.omega.common.utils;

public class CheckArrayUtils {
	
	public static boolean allCheck(float[][][][] x,float[][][][] y) {
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				for(int m = 0;m<x[i][j].length;m++) {
					for(int n = 0;n<x[i][j][m].length;n++) {
						if(x[i][j][m][n] != y[i][j][m][n]) {
							return false;
						}
					}
				}
			}
		}
		return true;
	}
	
	public static float check(float[][][][] x,float[][][][] y) {
		float error = 0.0f;
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				for(int m = 0;m<x[i][j].length;m++) {
					for(int n = 0;n<x[i][j][m].length;n++) {
						error += Math.abs((x[i][j][m][n] - y[i][j][m][n]));
					}
				}
			}
		}
		return error;
	}
	
	public static float check(float[][] x,float[][] y) {
		float error = 0.0f;
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				error += Math.abs((x[i][j] - y[i][j]));
			}
		}
		return error;
	}
	
	public static float check(float[] x,float[] y) {
		float error = 0.0f;
		for(int i = 0;i<x.length;i++) {
			error += Math.abs((x[i] - y[i]));
		}
		return error;
	}
	
}
