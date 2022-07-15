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
	public static float[] heRandom(int x,float n){
		float[] temp = new float[x];
		for(int i = 0;i<x;i++) {
			temp[i] = (float) (getInstance().nextGaussian() * Math.sqrt(2.0d / n));
		}
		return temp;
	}
	
	/**
	 * he随机数
	 * @param x
	 * @return
	 */
	public static float[][] heRandom(int x,int y,float n){
		float[][] temp = new float[x][y];
		float t = (float) Math.sqrt(2.0d / n);
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = (float) (Math.abs(getInstance().nextGaussian()) * t);
			}
		}
		return temp;
	}
	
	/**
	 * he随机数
	 * @param x
	 * @return
	 */
	public static float[][][][] heRandom(int c,int n,int x,int y,float nn){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) Math.sqrt(2.0d / nn);
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float)(getInstance().nextGaussian() * t);
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
	public static float[] gaussianRandom(int x,float ratio){
		float[] temp = new float[x];
		for(int i = 0;i<x;i++) {
			temp[i] = (float)(getInstance().nextGaussian() * ratio);
		}
		return temp;
	}
	
	/**
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static float[][] gaussianRandom(int x,int y,float ratio){
		float[][] temp = new float[x][y];
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = (float) (getInstance().nextGaussian() * ratio);
			}
		}
		return temp;
	}
	
	/**
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static float[][][][] gaussianRandom(int c,int n,int x,int y,float ratio){
		float[][][][] temp = new float[c][n][x][y];
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (getInstance().nextGaussian() * ratio);
					}
				}
			}
		}
		return temp;
	}
	
	public static float[] x2Random(int x) {
		float[] temp = new float[x];
		float scale = (float) Math.sqrt(1.0 / x);
        for(int i=0; i<x; i++) {
        	temp[i] = getInstance().nextFloat() * scale;
        }
        return temp;
	}
	
	public static float[][] x2Random(int x,int y){
		float[][] temp = new float[x][y];
		float scale = (float) Math.sqrt(1.0 / x);
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = getInstance().nextFloat() * scale;
			}
		}
		return temp;
	}
	
	public static float[][][][] x2Random(int c,int n,int x,int y) {
		float[][][][] temp = new float[c][n][x][y];
		float scale = (float) Math.sqrt(1.0 / x); 
        for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = getInstance().nextFloat() * scale;
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
	public static float[] xavierRandom(int x,float ratio){
		float[] temp = new float[x];
		for(int i = 0;i<x;i++) {
			temp[i] = (float) (getInstance().nextGaussian() / Math.sqrt(x));
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static float[][] xavierRandom(int x,int y,int fanIn,int fanOut){
		float[][] temp = new float[x][y];
		float t = (float) Math.sqrt(2.0f/(fanIn+fanOut));
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = (float) (getInstance().nextGaussian() * t);
			}
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static float[][] xavierRandomCaffe(int x,int y,int fanIn,int fanOut){
		float[][] temp = new float[x][y];
		float t = (float) 2.0f/(fanIn+fanOut);
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = (float) (getInstance().nextGaussian() * t);
			}
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static float[][][][] xavierRandom(int c,int n,int x,int y,int fanIn,int fanOut){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) Math.sqrt(2.0d/(fanIn+fanOut));
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
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
	public static float[][][][] heRandom(int c,int n,int x,int y,int fanIn){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) Math.sqrt(2.0d/fanIn);
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (Math.abs(getInstance().nextGaussian()) * t);
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
	public static float[][][][] xavierRandomCaffeIn(int c,int n,int x,int y,int fanIn,int fanOut){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) 1.0f / fanIn;
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
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
	public static float[][][][] xavierRandomCaffeOut(int c,int n,int x,int y,int fanIn,int fanOut){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) 1.0f / fanOut;
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
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
	public static float[][][][] xavierRandomCaffe(int c,int n,int x,int y,int fanIn,int fanOut){
		float[][][][] temp = new float[c][n][x][y];
		float t = (float) 2.0d / (fanIn+fanOut);
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
					}
				}
			}
		}
		return temp;
	}
	
}
