package com.omega.common.utils;

import java.util.Random;

/**
 * random utils
 * @author Administrator
 *
 */
public class RandomUtils {
	
	private static Random instance;
	
	public static Random getInstance() {
		if(instance == null) {
			instance = new Random();
		}
		return instance;
	}

	/**
	 * he随机数
	 * @param x
	 * @return
	 */
	public static double[] heRandom(int x,double n){
		double[] temp = new double[x];
		for(int i = 0;i<x;i++) {
			temp[i] = getInstance().nextGaussian() * Math.sqrt(2.0d / n);
		}
		return temp;
	}
	
	/**
	 * he随机数
	 * @param x
	 * @return
	 */
	public static double[][] heRandom(int x,int y,double n){
		double[][] temp = new double[x][y];
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = getInstance().nextGaussian() * Math.sqrt(2.0d / n);
			}
		}
		return temp;
	}
	
	/**
	 * he随机数
	 * @param x
	 * @return
	 */
	public static double[][][][] heRandom(int c,int n,int x,int y,double nn){
		double[][][][] temp = new double[c][n][x][y];
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = getInstance().nextGaussian() * Math.sqrt(2.0d / nn);
					}
				}
			}
		}
		return temp;
	}
	
	
	/**
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static double[] gaussianRandom(int x,double ratio){
		double[] temp = new double[x];
		for(int i = 0;i<x;i++) {
			temp[i] = getInstance().nextGaussian() * ratio;
		}
		return temp;
	}
	
	/**
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static double[][] gaussianRandom(int x,int y,double ratio){
		double[][] temp = new double[x][y];
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = getInstance().nextGaussian() * ratio;
			}
		}
		return temp;
	}
	
	/**
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static double[][][][] gaussianRandom(int c,int n,int x,int y,double ratio){
		double[][][][] temp = new double[c][n][x][y];
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = getInstance().nextGaussian() * ratio;
					}
				}
			}
		}
		return temp;
	}
	
	public static double[] x2Random(int x) {
		double[] temp = new double[x];
		double scale = Math.sqrt(1.0 / x);
        for(int i=0; i<x; i++) {
        	temp[i] = getInstance().nextDouble() * scale;
        }
        return temp;
	}
	
	public static double[][] x2Random(int x,int y){
		double[][] temp = new double[x][y];
		double scale = Math.sqrt(1.0 / x);
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = getInstance().nextDouble() * scale;
			}
		}
		return temp;
	}
	
	public static double[][][][] x2Random(int c,int n,int x,int y) {
		double[][][][] temp = new double[c][n][x][y];
		double scale = Math.sqrt(1.0 / x); 
        for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = getInstance().nextDouble() * scale;
					}
				}
			}
		}
        
        return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static double[] xavierRandom(int x,double ratio){
		double[] temp = new double[x];
		for(int i = 0;i<x;i++) {
			temp[i] = getInstance().nextGaussian() / Math.sqrt(x);
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static double[][] xavierRandom(int x,int y,int fanIn,int fanOut){
		double[][] temp = new double[x][y];
		double t = Math.sqrt(2.0d/(fanIn+fanOut));
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = getInstance().nextGaussian() * t;
			}
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static double[][][][] xavierRandom(int c,int n,int x,int y,int fanIn,int fanOut){
		double[][][][] temp = new double[c][n][x][y];
		double t = Math.sqrt(2.0d/(fanIn+fanOut));
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = getInstance().nextGaussian() * t;
					}
				}
			}
		}
		return temp;
	}
	
}
