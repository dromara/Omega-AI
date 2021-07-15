package com.omega.common.utils;

import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.engine.pooling.PoolingType;

/**
 * 
 * @ClassName: MatrixOperation
 *
 * @author lijiaming
 *
 * @date 2020年7月24日
 *
 * @Description: 
 * TODO(用一句话描述该文件做什么)
 *
 */
public class MatrixOperation {
	
	private static final int threadNum = 8;
	
	/**
	 * 
	 * @Title: exp
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] exp(double[] x) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = Math.exp(x[i]);
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: exp
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] pow(double[] x,double e) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = Math.pow(x[i],e);
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: exp
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] pow(double[][] x,double e) {
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = Math.pow(x[i][j],e);
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] add(double[] x,double b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] + b;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] add(double[][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] + b;
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] add(double[][][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				for(int n = 0;n<x[i][j].length;n++) {
					temp[i][j][n] = x[i][j][n] + b;
				}
			}
		}

		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] add(double[][][][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int h = 0;h<x[n][c].length;h++) {
					for(int w = 0;w<x[n][c][h].length;w++) {
						temp[n][c][h][w] = x[n][c][h][w] + b;
					}
				}
			}
		}
		return temp;
	}
	
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] add(double[] x,double[] b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] + b[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] add(double[][][][] x,double[][][][] b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int h = 0;h<x[n][c].length;h++) {
					for(int w = 0;w<x[n][c][h].length;w++) {
						temp[n][c][h][w] = x[n][c][h][w] + b[n][c][h][w];
					}
				}
			}
		}
		return temp;
	}
	

	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] add(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] + b[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] add(double[][][] x,double[] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != b.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i][j] = x[c][i][j] + b[c];
				}
			}
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] add(double[][][][] x,double[] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != b.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		
		for(int n = 0;n<x.length;n++) {

			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						temp[n][c][i][j] = x[n][c][i][j] + b[c];
					}
				}
			}
			
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @param type
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] addByBN(double[][][][] x,double[] b, int type) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != b.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		
		for(int n = 0;n<x.length;n++) {

			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						if(type == 0) {
							temp[n][c][i][j] = x[n][c][i][j] + b[j];
						}else {
							temp[n][c][i][j] = x[n][c][i][j] + b[c];
						}
					}
				}
			}
			
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: add
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] add(double[][][] x,double[][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != b.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i][j] = x[c][i][j] + b[c][i][j];
				}
			}
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] subtraction(double[] x,double b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] - b;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] subtractionForOne(double[][][] x,double b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i][0][0] - b;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] subtraction(double[][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] - b;
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtraction(double[][][][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int h = 0;h<x[n][c].length;h++) {
					for(int w = 0;w<x[n][c][h].length;w++) {
						temp[n][c][h][w] = x[n][c][h][w] - b;
					}
				}
			}
		}
		return temp;
	}
	

	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] subtraction(double[] x,double[] b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] - b[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] subtractionP(double[] x,double[] b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] - b[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] subtractionP(double[][] x,double[][] b) {
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				temp[m][i] = x[m][i] - b[m][i];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtractionP(double[][][][] x,double[][][][] b) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						temp[m][i][j][n] = x[m][i][j][n] - b[m][i][j][n];
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtractionP(double[][][][] x,double[][][] b) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						temp[m][i][j][n] = x[m][i][j][n] - b[i][j][n];
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @param type 0:fully,1:conv
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtraction(double[][][][] x,double[] b,int type) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						if(type == 0) {
							temp[m][i][j][n] = x[m][i][j][n] - b[n];
						}else {
							temp[m][i][j][n] = x[m][i][j][n] - b[i];
						}
						
					}
				}
			}
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtractionForConv(double[][][][] x,double[] b) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						temp[m][i][j][n] = x[m][i][j][n] - b[i];
					}
				}
			}
		}
		
		return temp;
	}
	

	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] subtraction(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] - b[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] subtraction(double[][][] x,double[][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					temp[k][i][j] = x[k][i][j] - b[k][i][j];
				}
			}
		}

		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] subtraction(double[][][][] x,double[][][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int k = 0;k<x[c].length;k++) {
				for(int i = 0;i<x[c][k].length;i++) {
					for(int j = 0;j<x[c][k][i].length;j++) {
						temp[c][k][i][j] = x[c][k][i][j] - b[c][k][i][j];
					}
				}
			}
		}
		
		return temp;
	}
	
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] multiplication(double[] x,double b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] * b;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] multiplication(double[][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] * b;
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] multiplication(double[][][] x,double b) {
		
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i][j] = x[c][i][j] * b;
				}
			}
			
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] multiplication(double[][][][] x,double b) {
		
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int k = 0;k<x[c].length;k++) {
				for(int i = 0;i<x[c][k].length;i++) {
					for(int j = 0;j<x[c][k][i].length;j++) {
						temp[c][k][i][j] = x[c][k][i][j] * b;
					}
				}
			}
			
		}
		
		return temp;
	}

	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] multiplication(double[] x,double[] b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] * b[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @param type 0:fully,1:conv
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] multiplicationByBN(double[][][][] x,double[] gama,int type) {
		if(x == null || gama == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != gama.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						if(type == 0) {
							temp[n][c][i][j] = x[n][c][i][j] * gama[j];
						}else {
							temp[n][c][i][j] = x[n][c][i][j] * gama[c];
						}
					}
				}
			}
		}
		
		return temp;
	}
	

	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] multiplication(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] * b[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] multiplicationForMatrix(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixUtils.zero(x.length,b[0].length);
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < b[0].length; j++){
				for(int k = 0; k < x[0].length; k++){
					temp[i][j] += x[i][k] * b[k][j];
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] multiplicationByEjml(double[][] x,double[][] z) {
		double[][] temp = MatrixUtils.zero(x.length,z[0].length);
		SimpleMatrix a = new SimpleMatrix(x);
		SimpleMatrix b = new SimpleMatrix(z);
		SimpleMatrix c = a.mult(b);
		for(int i = 0;i<temp.length;i++) {
			for(int j = 0;j<temp[i].length;j++) {
				temp[i][j] = c.get(i, j);
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] multiplication(double[][][] x,double[][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length || x[0][0].length != b[0][0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i][j] = x[c][i][j] * b[c][i][j];
				}
			}
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: multiplication
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] multiplication(double[][][][] x,double[][][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length || x[0][0].length != b[0][0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					for(int l = 0;l<x[c][i][j].length;l++) {
						temp[c][i][j][l] = x[c][i][j][l] * b[c][i][j][l];
					}
				}
			}
		}
		
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] division(double[] x,double b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] / b;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] division(double[][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] / b;
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] division(double[][][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int m = 0;m<x[n].length;m++) {
				for(int i = 0;i<x[n][m].length;i++) {
					temp[n][m][i] = x[n][m][i] / b;
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] division(double[][][][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int m = 0;m<x[n].length;m++) {
				for(int i = 0;i<x[n][m].length;i++) {
					for(int j = 0;j<x[n][m][i].length;j++) {
						temp[n][m][i][j] = x[n][m][i][j] / b;
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @param type 0:fully,1:conv
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] division(double[][][][] x,double b[],int type) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int n = 0;n<x.length;n++) {
			for(int m = 0;m<x[n].length;m++) {
				for(int i = 0;i<x[n][m].length;i++) {
					for(int j = 0;j<x[n][m][i].length;j++) {
						if(type == 0) {
							temp[n][m][i][j] = x[n][m][i][j] / b[j];
						}else {
							temp[n][m][i][j] = x[n][m][i][j] / b[m];
						}
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] division(double[] x,double[] b) {
		double[] temp = MatrixUtils.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] / b[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: division
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] division(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixUtils.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[0].length;j++) {
				temp[i][j] = x[i][j] / b[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] division(double[][][][] x,double[][][] b) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						temp[m][i][j][n] = x[m][i][j][n] / b[i][j][n];
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: subtraction
	 *
	 * @param x
	 * @param b
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] division(double[][][][] x,double[][][][] b) {
		double[][][][] temp = MatrixUtils.zero(x.length,x[0].length,x[0][0].length,x[0][0][0].length);
		for(int m = 0;m<x.length;m++) {
			for(int i = 0;i<x[m].length;i++) {
				for(int j = 0;j<x[m][i].length;j++) {
					for(int n = 0;n<x[m][i][j].length;n++) {
						temp[m][i][j][n] = x[m][i][j][n] / b[m][i][j][n];
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double sum(double[] x) {
		double temp = 0.0d;
		for(double o:x) {
			temp += o;
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double sum(double[][] x) {
		double temp = 0.0d;
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp += x[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double sum(double[][][][] x) {
		double temp = 0.0d;
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				for(int n = 0;n<x[i][j].length;n++) {
					for(int m = 0;m<x[i][j][n].length;m++) {
						temp += x[i][j][n][m];
					}
				}
			}
			
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] sumByBn(double[][][][] x) {
		double[][][] temp = new double[x[0].length][x[0][0].length][x[0][0][0].length];
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				for(int n = 0;n<x[i][j].length;n++) {
					for(int m = 0;m<x[i][j][n].length;m++) {
						temp[j][n][m] += x[i][j][n][m];
					}
				}
			}
			
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] sumBias(double[][][] x) {
		double[] temp = new double[x.length];
		for(int i = 0;i<x.length;i++) {
			for(int h = 0;h<x[i].length;h++) {
				for(int w = 0;w<x[i][h].length;w++) {
					temp[i] += x[i][h][w];
				}
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: count
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] sumBias(double[][][][] x) {
		double[] temp = new double[x[0].length];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int h = 0;h<x[n][c].length;h++) {
					for(int w = 0;w<x[n][c][h].length;w++) {
						temp[c] += x[n][c][h][w];
					}
				}
			}
		}
		return temp;
	}
	
	
	/**
	 * isZero
	 * @param x
	 * @return
	 */
	public static boolean isZero(double[][][] x) {
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					if(x[k][i][j] != 0) {
						return false;
					}
				}
			}
		}
		return true;
	}
	
	/**
	 * isZero
	 * @param x
	 * @return
	 */
	public static boolean isZero(double[] x) {
		for(int k = 0;k<x.length;k++) {
			if(x[k] != 0) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * 
	 * @Title: max
	 *
	 * @return
	 *
	 * @Description:
	 *
	 * @throws
	 */
	public static double max(double[] x) {
		double max = 0.0d;
		if(x.length > 0) {
			max = x[0];
			for(int i= 0;i<x.length;i++) {
				if(max <= x[i]) {
					max = x[i];
				}
			}
		}
		return max;
	}
	
	/**
	 * 
	 * @Title: max
	 *
	 * @return
	 *
	 * @Description:
	 *
	 * @throws
	 */
	public static double max(double[][][] x) {
		double max = 0.0d;
		if(x.length > 0) {
			max = x[0][0][0];
			for(int c= 0;c<x.length;c++) {
				for(int h = 0;h<x[c].length;h++) {
					for(int w = 0;w<x[c][h].length;w++) {
						if(max <= x[c][h][w]) {
							max = x[c][h][w];
						}
					}
				}
			}
		}
		return max;
	}
	
	/**
	 * 
	 * @Title: maxIndex
	 *
	 * @return
	 *
	 * @Description:
	 *
	 * @throws
	 */
	public static int maxIndex(double[] x) {
		int index = 0;
		if(x.length > 0) {
			double max = x[0];
			for(int i= 0;i<x.length;i++) {
				if(max <= x[i]) {
					max = x[i];
					index = i;
				}
			}
		}
		return index;
	}
	
	/**
	 * 
	 * @Title: maxIndex
	 *
	 * @return
	 *
	 * @Description:
	 *
	 * @throws
	 */
	public static int maxIndex(double[][][] x) {
		int index = 0;
		if(x.length > 0) {
			double max = x[0][0][0];
			for(int i= 0;i<x.length;i++) {
				if(max <= x[i][0][0]) {
					max = x[i][0][0];
					index = i;
				}
			}
		}
		return index;
	}
	
	/**
	 * mean
	 * @param x
	 * @return
	 */
	public static double[][][] mean(double[][][][] x){
		
		double[][][] mean = new double[x[0].length][x[0][0].length][x[0][0][0].length];
		
		for(int m = 0;m<x.length;m++) {
			for(int c = 0;c<x[m].length;c++) {
				for(int h = 0;h<x[m][c].length;h++) {
					for(int  w = 0;w<x[m][c][h].length;w++) {
						mean[c][h][w] += x[m][c][h][w] / x.length;
					}
				}
			}
		}
		
		return mean;
	}
	
	/**
	 * mean
	 * @param x
	 * @param type 0:fully,1:conv
	 * @return
	 */
	public static double[] mean(double[][][][] x,int type){
		
		int count = 0;

		if(type == 0) {
			count = x[0][0][0].length;
		}else {
			count = x[0].length;
		}
		
		double[] mean = new double[count];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int m = 0;m<x.length;m++) {
			final int index = m;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<x[index].length;c++) {
						for(int h = 0;h<x[index][c].length;h++) {
							for(int  w = 0;w<x[index][c][h].length;w++) {
								if(type == 0) {
									mean[w] += x[index][c][h][w] / x.length;
								}else {
									mean[c] += x[index][c][h][w] / x.length / x[0][0].length / x[0][0][0].length;
								}
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return mean;
	}
	
	/**
	 * standard deviation
	 * @param x
	 * @return
	 */
	public static double[][][] std(double[][][][] x){
		
		double[][][] std = new double[x[0].length][x[0][0].length][x[0][0][0].length];
		
		std = MatrixOperation.var(x);
		
		for(int c = 0;c<std.length;c++) {
			for(int h = 0;h<std[c].length;h++) {
				for(int  w = 0;w<std[c][h].length;w++) {
					std[c][h][w] = Math.sqrt(std[c][h][w]);
				}
			}
		}

		return std;
	}
	
	/**
	 * standard deviation
	 * @param x
	 * @param type 0:fully,1:conv
	 * @return
	 */
	public static double[] std(double[][][][] x,int eta,int type){
		
		double[] std = MatrixOperation.var(x, type);
		
		for(int c = 0;c<std.length;c++) {
			std[c] = Math.sqrt(std[c] + eta);
		}

		return std;
	}
	
	/**
	 * standard deviation
	 * @param x
	 * @param type 0:fully,1:conv
	 * @return
	 */
	public static double[] std(double[] x){
		
		double[] std = new double[x.length];
		
		for(int c = 0;c<std.length;c++) {
			std[c] = Math.sqrt(x[c]);
		}

		return std;
	}
	
	/**
	 * standard deviation
	 * @param x
	 * @return
	 */
	public static double[][][] var(double[][][][] x){
		
		double[][][] mean = MatrixOperation.mean(x);
		
		double[][][] std = new double[x[0].length][x[0][0].length][x[0][0][0].length];
		
		for(int m = 0;m<x.length;m++) {
			for(int c = 0;c<x[m].length;c++) {
				for(int h = 0;h<x[m][c].length;h++) {
					for(int  w = 0;w<x[m][c][h].length;w++) {
						std[c][h][w] += ((x[m][c][h][w] - mean[c][h][w]) * (x[m][c][h][w] - mean[c][h][w])) / x.length;
					}
				}
			}
		}
		
		return std;
	}
	
	/**
	 * standard deviation
	 * @param x
	 * @param type 0:fully,1:conv
	 * @return
	 */
	public static double[] var(double[][][][] x, int type){
		
		double[] mean = MatrixOperation.mean(x, type);
		
		double[] std = new double[mean.length];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int m = 0;m<x.length;m++) {
			final int index = m;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<x[index].length;c++) {
						for(int h = 0;h<x[index][c].length;h++) {
							for(int  w = 0;w<x[index][c][h].length;w++) {
								if(type == 0) {
									std[w] += ((x[index][c][h][w] - mean[w]) * (x[index][c][h][w] - mean[w])) / x.length;
								}else {
									std[c] += ((x[index][c][h][w] - mean[c]) * (x[index][c][h][w] - mean[c])) / x.length;
								}
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return std;
	}
	
	/**
	 * pow
	 * @return
	 */
	public static double[][][] pow(double[][][] x,double exponent){
		
		double[][][] y = new double[x.length][x[0].length][x[0][0].length];
		
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					y[k][i][j] = Math.pow(x[k][i][j], exponent);
				}
			}
		}
		
		return y;
	}
	
	/**
	 * pow
	 * @return
	 */
	public static double[][][][] pow(double[][][][] x,double exponent){
		
		double[][][][] y = new double[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					for(int l = 0;l<x[k][i][j].length;l++) {
						y[k][i][j][l] = Math.pow(x[k][i][j][l], exponent);
					}
				}
			}
		}
		
		return y;
	}
	
	/**
	 * sqrt
	 * @return
	 */
	public static double[] sqrt(double[] x){
		
		double[] y = new double[x.length];
		
		for(int i = 0;i<x.length;i++) {
			y[i] = Math.sqrt(x[i]);
		}

		return y;
	}
	
	/**
	 * sqrt
	 * @return
	 */
	public static double[][] sqrt(double[][] x){
		
		double[][] y = new double[x.length][x[0].length];
		
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				y[i][j] = Math.sqrt(x[i][j]);
			}
		}

		return y;
	}
	
	/**
	 * sqrt
	 * @return
	 */
	public static double[][][] sqrt(double[][][] x){
		
		double[][][] y = new double[x.length][x[0].length][x[0][0].length];
		
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					y[k][i][j] = Math.sqrt(x[k][i][j]);
				}
			}
		}
		
		return y;
	}
	
	/**
	 * sqrt
	 * @return
	 */
	public static double[][][][] sqrt(double[][][][] x){
		
		double[][][][] y = new double[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
		
		for(int k = 0;k<x.length;k++) {
			for(int i = 0;i<x[k].length;i++) {
				for(int j = 0;j<x[k][i].length;j++) {
					for(int l = 0;l<x[k][i][j].length;l++) {
						y[k][i][j][l] = Math.sqrt(x[k][i][j][l]);
					}
				}
			}
		}
		
		return y;
	}
	
	
	/**
	 * rotate90
	 * @param matrix
	 * @return
	 */
	public static double[][] rotate90(double[][] matrix){
		double[][] temp = new double[matrix[0].length][matrix.length];
		int dst=matrix.length-1;
		for(int i=0;i<matrix.length;i++,dst--){
			for(int j=0;j<matrix[i].length;j++){
				temp[j][dst]=matrix[i][j];
			}
		}
		return temp;
	 }

	/**
	 * rotate90
	 * @param matrix
	 * @return
	 */
	public static double[][][][] rotate90(double[][][][] matrix){
		double[][][][] temp = new double[matrix.length][matrix[0].length][matrix[0][0][0].length][matrix[0][0].length];
		for(int c=0;c<matrix.length;c++) {
			for(int k=0;k<matrix[c].length;k++) {
				int dst=matrix[0][0].length-1;
				for(int i=0;i<matrix[c][k].length;i++,dst--){
					for(int j=0;j<matrix[c][k][i].length;j++){
						temp[c][k][j][dst]=matrix[c][k][i][j];
					}
				}
			}
		}
		
		return temp;
	 }

	/**
	 * revolve180
	 * @param x
	 * @return
	 */
	public static double[][] rotate180(double[][] x){
		
		double[][] temp = MatrixOperation.rotate90(x);
		
		temp = MatrixOperation.rotate90(temp);
		
		return temp;
	}
	
	/**
	 * revolve180
	 * @param x
	 * @return
	 */
	public static double[][][][] rotate180(double[][][][] x){
		
		double[][][][] temp = MatrixOperation.rotate90(x);
		
		temp = MatrixOperation.rotate90(temp);
		
		return temp;
	}

	/**
	 * zeroPadding
	 * @param x
	 * @param paddingNum
	 * @return x + padding * 2
	 */
	public static double[][] zeroPadding(double[][] x,int paddingNum){
		double[][] temp = new double[x.length + paddingNum * 2][x[0].length + paddingNum * 2];
		
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i+paddingNum][j+paddingNum] = x[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * zeroPadding
	 * @param x
	 * @param paddingNum
	 * @return x + padding * 2
	 */
	public static double[][][] zeroPadding(double[][][] x,int paddingNum){
		double[][][] temp = new double[x.length][x[0].length + paddingNum * 2][x[0][0].length + paddingNum * 2];
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i+paddingNum][j+paddingNum] = x[c][i][j];
				}
			}
		}
		return temp;
	}
	
	/**
	 * zeroPadding
	 * @param x
	 * @param paddingNum
	 * @return x + padding * 2
	 */
	public static double[][][][] zeroPadding(double[][][][] x,int paddingNum){
		double[][][][] temp = new double[x.length][x[0].length][x[0][0].length + paddingNum * 2][x[0][0][0].length + paddingNum * 2];
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int i = 0;i<x[n][c].length;i++) {
					for(int j = 0;j<x[n][c][i].length;j++) {
						temp[n][c][i+paddingNum][j+paddingNum] = x[n][c][i][j];
					}
				}
			}
		}
		return temp;
	}
	
	/**
	 * dislodgeZeroPadding
	 * @param x
	 * @param paddingNum
	 * @return x - padding * 2
	 */
	public static double[][] dislodgeZeroPadding(double[][] x,int paddingNum){
		double[][] temp = new double[x.length -+ paddingNum * 2][x[0].length -+ paddingNum * 2];
		
		for(int i = 0;i<temp.length;i++) {
			for(int j = 0;j<temp[i].length;j++) {
				temp[i][j] = x[i+paddingNum][j+paddingNum];
			}
		}
		
		return temp;
	}
	
	/**
	 * dislodgeZeroPadding
	 * @param x
	 * @param paddingNum
	 * @return x - padding * 2
	 */
	public static double[][][] dislodgeZeroPadding(double[][][] x,int paddingNum){
		double[][][] temp = new double[x.length][x[0].length - paddingNum * 2][x[0][0].length - paddingNum * 2];
		for(int c = 0;c<temp.length;c++) {
			for(int i = 0;i<temp[c].length;i++) {
				for(int j = 0;j<temp[c][i].length;j++) {
					temp[c][i][j] = x[c][i+paddingNum][j+paddingNum];
				}
			}
		}
		return temp;
	}
	
	/**
	 * convn full
	 * @param x
	 * @param k
	 * @return
	 * 
	 * zeroPadding = (K - 1) / 2
	 * o = (W -K + 2P) / S + 1
	 */
	public static double[][] convnFull(double[][] x,double[][] k,int stride){
		
		int zeroPadding = k.length - 1;
		int outputSize = ((x.length - k.length + 2 * zeroPadding) / stride) + 1;
		
		double[][] result = new double[outputSize][outputSize];
		
		int kernelSize = k.length - 1;
		
		for(int i = 0;i<outputSize;i++) {
			
			for(int j = 0;j<outputSize;j++) {
				
				result[i][j] = 0.0d;
				
				for(int m = 0;m<k.length;m++) {
					
					for(int n = 0;n<k[m].length;n++) {
						
						if((i - kernelSize + m) < 0) {
							result[i][j] += 0.0d;
						}else if((j - kernelSize + n) < 0) {
							result[i][j] += 0.0d;
						}else if((i - kernelSize + m) >=  x.length) {
							result[i][j] += 0.0d;
						}else if((j - kernelSize + n) >= x.length) {
							result[i][j] += 0.0d;
						}else {
							result[i][j] += x[i - kernelSize + m][j - kernelSize + n] * k[m][n];
						}

					}
					
				}
				
			}
			
		}
		
		return result;
	}
	
	/**
	 * convn full
	 * @param x
	 * @param k
	 * @return
	 * 
	 * zeroPadding = (K - 1) / 2
	 * o = (W -K + 2P) / S + 1
	 */
	public static double[][] convnSame(double[][] x,double[][] k,int stride){
		
		int zeroPadding = (k.length - 1) / 2;
		int outputSize = ((x.length - k.length + 2 * zeroPadding) / stride) + 1;

		double[][] result = new double[outputSize][outputSize];
		
		int kernelSize = ((k.length + 1) / 2) - 1;
		
		for(int i = 0;i<outputSize;i++) {
			
			for(int j = 0;j<outputSize;j++) {
				
				result[i][j] = 0.0d;
				
				for(int m = 0;m<k.length;m++) {
					
					for(int n = 0;n<k[m].length;n++) {

						if((i - kernelSize + m) < 0) {
							result[i][j] += 0.0d;
						}else if((j - kernelSize + n) < 0) {
							result[i][j] += 0.0d;
						}else if((i - kernelSize + m) >= outputSize) {
							result[i][j] += 0.0d;
						}else if((j - kernelSize + n) >= outputSize) {
							result[i][j] += 0.0d;
						}else {
							result[i][j] += x[i - kernelSize + m][j - kernelSize + n] * k[m][n];
						}

					}
					
				}
				
			}
			
		}
		
		return result;
	}
	
	/**
	 * convnVail
	 * @param x  h * w
	 * @param k  kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][] convnVail(double[][] x,double[][] k,int stride){
		
		int oHeight = ((x.length - k.length ) / stride) + 1;
		
		int oWidth = ((x[0].length - k[0].length) / stride) + 1;

		double[][] result = new double[oHeight][oWidth];
		
		for(int i = 0;i<oHeight;i++) {
			
			for(int j = 0;j<oWidth;j++) {
				
				for(int m = 0;m<k.length;m++) {
					
					for(int n = 0;n<k[m].length;n++) {

						result[i][j] += x[i * stride + m][j * stride + n] * k[m][n];

					}
					
				}
				
			}
			
		}
		
		return result;
	}
	
	/**
	 * convnVail
	 * @param x  c * h * w
	 * @param k  c * kn * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][] convnVail(double[][][] x,double[][][][] k,int stride){
		
		int channel = x.length;
		
		int kNum = k[0].length;
		
		int oHeight = ((x[0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][] result = new double[kNum][oHeight][oWidth];

		for(int kn = 0;kn<kNum;kn++) {
			
			for(int i = 0;i<oHeight;i++) {
				
				for(int j = 0;j<oWidth;j++) {
					
					for(int c = 0;c<channel;c++) {
						
						for(int m = 0;m<k[c][kn].length;m++) {
							
							for(int n = 0;n<k[c][kn][m].length;n++) {
								
								result[kn][i][j] += x[c][i * stride + m][j * stride + n] * k[c][kn][m][n];

							}
							
						}
						
					}
					
				}
				
			}
			
		}

		return result;
	}
	
	/**
	 * im2col2
	 * @param x
	 * @param kh
	 * @param kw
	 * @param stride
	 * @return
	 */
	public static double[][] im2col2(double[][][] x,int kh,int kw,int stride){
		
		int oHeight = ((x[0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - kw) / stride) + 1;
		
		int ow = x.length * kh * kw;
		
		int oh = oHeight * oWidth;
		
		int kSize = kh * kw;
		
		double[][] result = new double[oh][ow];
		
		for(int i = 0;i<oh;i++) {
			
			int startH = i / oHeight * stride;
			
			int startW = i % oWidth * stride;
			
			for(int j = 0;j<ow;j++) {
				
				int c = j / kSize;
				
				int xSize = j - (c * kSize);
				
				int xh = startH + xSize / kw;
				
				int xw = startW + xSize % kw;
				
				result[i][j] = x[c][xh][xw];

			}
			
		}
		
		return result;
	}
	
	/**
	 * im2col
	 * @param x
	 * @param kh
	 * @param kw
	 * @param stride
	 * @return
	 */
	public static double[][] im2col(double[][][] x,int kh,int kw,int stride){
		
		int oHeight = ((x[0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - kw) / stride) + 1;
		
		int channel = x.length;
		
		double[][] result = new double[oHeight * oWidth][channel * kh * kw];
		
		int hIndex = 0;
		
		for(int oh = 0;oh<oHeight;oh++) {
			
			for(int ow = 0;ow<oWidth;ow++) {

				int wIndex = 0;
				
				for(int c = 0;c<channel;c++) {

					for(int h = 0;h<kh;h++) {
						
						for(int w = 0;w<kw;w++) {
							
							result[hIndex][wIndex] = x[c][oh * stride + h][ow * stride + w];
							
							wIndex++;
						}
						
					}
					
				}

				hIndex++;
			}
			
		}
		
		return result;
	}
	
	/**
	 * im2col
	 * @param x
	 * @param kh
	 * @param kw
	 * @param stride
	 * @return
	 */
	public static double[][] kernel2col(double[][][][] k,int kh,int kw){
		
		int ic = k.length;
				
		int oc = k[0].length;
		
		double[][] result = new double[ic * kh * kw][oc];
		
		int hIndex = 0;
		
		for(int c = 0;c<ic;c++) {
			
			for(int h = 0;h<kh;h++) {
				
				for(int w = 0;w<kw;w++) {
					
					for(int n = 0;n<oc;n++) {
						
						result[hIndex][n] = k[c][n][h][w];

					}
					hIndex++;
				}
				
			}
			
		}

		return result;
	}
	
	/**
	 * im2col
	 * @param x
	 * @param kh
	 * @param kw
	 * @param stride
	 * @return
	 */
	public static double[][] kernel2colToBack(double[][][][] k,int kh,int kw){
		
		int ic = k.length;
				
		int oc = k[0].length;
		
		double[][] result = new double[oc * kh * kw][ic];
		
		int hIndex = 0;
		
		for(int n = 0;n<oc;n++) {
			
			for(int h = 0;h<kh;h++) {
				
				for(int w = 0;w<kw;w++) {
					
					for(int c = 0;c<ic;c++) {
						
						result[hIndex][c] = k[c][n][h][w];

					}
					hIndex++;
				}
				
			}
			
		}

		return result;
	}
	
	public static double[][][][] convnVailByIm2Col(double[][][][] x,double[][][][] k,int stride,boolean isBack){
		
		int ko = k[0].length;
		
		if(isBack) {
			ko = k.length;
		}
		
		int kh = k[0][0].length;
		
		int kw = k[0][0][0].length;
		
		int oHeight = ((x[0][0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][][] result = new double[x.length][ko][oHeight][oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int bn = 0;bn<x.length;bn++) {
			
			final int index = bn;
			
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {

					double[][] col = MatrixOperation.im2col(x[index], kh, kw, stride);
					
					double[][] colK = MatrixOperation.kernel2col(k, kh, kw);
					
					if(isBack) {
						colK = MatrixOperation.kernel2colToBack(k, kh, kw);
					}

//					double[][] output = MatrixOperation.multiplicationForMatrix(col, colK);

					double[][] output = MatrixOperation.multiplicationByEjml(col, colK);
					
					result[index] = MatrixUtils.transform(output, oHeight, oWidth);

					return null;
				}
			});
			
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return result;
	}
	
	public static double[][][] convnVailByIm2Col(double[][][] x,double[][][][] k,int stride,boolean isBack){
		
		int ko = k[0].length;
		
		if(isBack) {
			ko = k.length;
		}
		
		int kh = k[0][0].length;
		
		int kw = k[0][0][0].length;
		
		int oHeight = ((x[0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][] result = new double[ko][oHeight][oWidth];

		double[][] col = MatrixOperation.im2col(x, kh, kw, stride);
		
		double[][] colK = MatrixOperation.kernel2col(k, kh, kw);
		
		if(isBack) {
			colK = MatrixOperation.kernel2colToBack(k, kh, kw);
		}

		double[][] output = MatrixOperation.multiplicationForMatrix(col, colK);
		
		result = MatrixUtils.transform(output, oHeight, oWidth);

		return result;
	}
	
	/**
	 * convnVail
	 * @param x  c * h * w
	 * @param k  c * kn * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][][] convnVail(double[][][][] x,double[][][][] k,int stride){
		
		int channel = x[0].length;
		
		int kNum = k[0].length;
		
		int oHeight = ((x[0][0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][][] result = new double[x.length][kNum][oHeight][oWidth];
		
		for(int b = 0;b<x.length;b++) {
			double[][][] bo = new double[kNum][oHeight][oWidth];
			for(int kn = 0;kn<kNum;kn++) {

				for(int c = 0;c<channel;c++) {
					
					for(int i = 0;i<oHeight;i++) {
					
						for(int j = 0;j<oWidth;j++) {
						
							for(int m = 0;m<k[c][kn].length;m++) {
								
								for(int n = 0;n<k[c][kn][m].length;n++) {
									
									bo[kn][i][j] += x[b][c][i * stride + m][j * stride + n] * k[c][kn][m][n];
									
								}
								
							}
							
						}
						
					}
					
				}
				
			}
			result[b] = bo;
		}

		return result;
	}
	
	/**
	 * convnVailForBack
	 * @param x  c * h * w
	 * @param k  c * kn * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][][] convnVailForBack(double[][][][] k,double[][][][] x,int stride){
		
		int channel = k.length;
		
		int kNum = k[0].length;
		
		int oHeight = ((x[0][0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][][] diff = new double[x.length][channel][oHeight][oWidth];

		for(int b = 0;b<x.length;b++) {

			for(int c = 0;c<channel;c++) {
				
				for(int kn = 0;kn<kNum;kn++) {
					
					for(int i = 0;i<oHeight;i++) {
						
						for(int j = 0;j<oWidth;j++) {
							
							for(int m = 0;m<k[c][kn].length;m++) {
									
								for(int n = 0;n<k[c][kn][m].length;n++) {
										
									diff[b][c][i][j] += x[b][kn][i * stride + m][j * stride + n] * k[c][kn][m][n];
		
								}
									
							}
								
						}
							
					}
				}
			}
			
		}
		
		return diff;
	}
	
	/**
	 * convnVail
	 * @param x  c * h * w
	 * @param k  kn * kc * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][][] convnVail(double[][][] x,double[][][] d,int stride){
		
		int channel = x.length;
		
		int kNum = d.length;
		
		int oHeight = ((x[0].length - d[0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - d[0][0].length) / stride) + 1;
		
		double[][][][] result = new double[channel][kNum][oHeight][oWidth];

		for(int c = 0;c<channel;c++) {
			
			for(int kn = 0;kn<kNum;kn++) {
				
				for(int i = 0;i<oHeight;i++) {
					
					for(int j = 0;j<oWidth;j++) {
						
						for(int m = 0;m<d[kn].length;m++) {
								
							for(int n = 0;n<d[kn][m].length;n++) {
								
								result[c][kn][i][j] += x[c][i * stride + m][j * stride + n] * d[kn][m][n];

							}
							
						}
						
					}
						
				}
					
			}
			
		}
		
		return result;
	}
	
	/**
	 * convnVailForDelta
	 * @param x  c * h * w
	 * @param k  kn * kc * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][][] convnVailForDelta(double[][][][] x,double[][][][] d,int stride){
		
		int channel = x[0].length;
		
		int kNum = d[0].length;
		
		int oHeight = ((x[0][0].length - d[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - d[0][0][0].length) / stride) + 1;
		
		double[][][][] result = new double[channel][kNum][oHeight][oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int b = 0;b<x.length;b++) {
			
			int index = b;
			
			workers.add(new Task<Object>(index) {
				
				@Override
			    public Object call() throws Exception {

					for(int c = 0;c<channel;c++) {
						
						for(int kn = 0;kn<kNum;kn++) {
							
							for(int i = 0;i<oHeight;i++) {
								
								for(int j = 0;j<oWidth;j++) {
									
									for(int m = 0;m<d[index][kn].length;m++) {
											
										for(int n = 0;n<d[index][kn][m].length;n++) {
											
											result[c][kn][i][j] += x[index][c][i * stride + m][j * stride + n] * d[index][kn][m][n];

										}
										
									}
									
								}
									
							}
								
						}
						
					}

					return null;
				}
			
			});
			
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return result;
	}
	
	/**
	 * pooling
	 * @param x
	 * @param pWidth
	 * @param pHeight
	 * @param stride
	 * @param poolingType
	 * @return
	 * 
	 * o = (W - ps) / S + 1
	 */
	public static double[][][] pooling(double[][][] x,int pWidth,int pHeight,int stride,PoolingType poolingType){
		
		int channel = x.length;
		
		int oHeight = ((x[0].length - pHeight ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - pWidth) / stride) + 1;

		double[][][] result = new double[channel][oHeight][oWidth];
		
		for(int c = 0;c<channel;c++) {
			
			for(int i = 0;i<oHeight;i++) {
				
				for(int j = 0;j<oWidth;j++) {
					
					for(int m = 0;m<pHeight;m++) {
						
						for(int n = 0;n<pWidth;n++) {
							
							switch (poolingType) {
							case MAX_POOLING:
								
								if(m == 0 && n == 0) {
									result[c][i][j] = x[c][i * stride + m][j * stride + n];
								}else if(result[c][i][j] < x[c][i * stride + m][j * stride + n]) {
									result[c][i][j] = x[c][i * stride + m][j * stride + n];
								}

								break;
							case MEAN_POOLING:
								
								result[c][i][j] += x[c][i * stride + m][j * stride + n];
								
								break;
							}
							
						}
						
					}
					
					switch (poolingType) {
					case MAX_POOLING:
						
						break;
					case MEAN_POOLING:
						
						result[c][i][j] /= pWidth * pHeight;
						
						break;
					}
					
				}
				
			}
			
		}

		return result;
	}
	
	/**
	 * pooling
	 * @param x
	 * @param pWidth
	 * @param pHeight
	 * @param stride
	 * @param poolingType
	 * @return
	 * 
	 * o = (W - ps) / S + 1
	 */
	public static double[][][] poolingAndMask(double[][][] x,double[][][][] mask,int pWidth,int pHeight,int stride,PoolingType poolingType){
		
		int channel = x.length;
		
		int oHeight = ((x[0].length - pHeight ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - pWidth) / stride) + 1;

		double[][][] result = new double[channel][oHeight][oWidth];
		
		for(int c = 0;c<channel;c++) {
			
			int maskIndex = 0;
			
			for(int i = 0;i<oHeight;i++) {
				
				for(int j = 0;j<oWidth;j++) {
					
					int maxH = 0;
					
					int maxW = 0;
					
					for(int m = 0;m<pHeight;m++) {
						
						for(int n = 0;n<pWidth;n++) {
							
							switch (poolingType) {
							case MAX_POOLING:
								
								if(m == 0 && n == 0) {
									result[c][i][j] = x[c][i * stride + m][j * stride + n];
								}else if(result[c][i][j] <= x[c][i * stride + m][j * stride + n]) {
									result[c][i][j] = x[c][i * stride + m][j * stride + n];
									maxH = m;
									maxW = n;
								}

								break;
							case MEAN_POOLING:
								result[c][i][j] += x[c][i * stride + m][j * stride + n];
								mask[c][maskIndex][m][n] = 1.0d / pWidth / pHeight;
								break;
							}
							
						}
						
					}
					
					switch (poolingType) {
					case MAX_POOLING:
						mask[c][maskIndex][maxH][maxW] = 1;
						break;
					case MEAN_POOLING:
						result[c][i][j] /= pWidth * pHeight;
						break;
					}
					
					maskIndex++;
				}
				
			}
			
		}

		return result;
	}
	
	/**
	 * pooling
	 * @param x
	 * @param pWidth
	 * @param pHeight
	 * @param stride
	 * @param poolingType
	 * @return
	 * 
	 * o = (W - ps) / S + 1
	 */
	public static double[][][][] poolingAndMask(double[][][][] x,double[][][][][] mask,int pWidth,int pHeight,int stride,PoolingType poolingType){
		
		int number = x.length;
		
		int channel = x[0].length;
		
		int oHeight = ((x[0][0].length - pHeight ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - pWidth) / stride) + 1;

		double[][][][] result = new double[number][channel][oHeight][oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int b = 0;b<number;b++) {
			
			final int index = b;
			
			workers.add(new Task<Object>(index) {
				
				@Override
				public Object call() {
					
					for(int c = 0;c<channel;c++) {
						
						int maskIndex = 0;
						
						for(int i = 0;i<oHeight;i++) {
							
							for(int j = 0;j<oWidth;j++) {
								
								int maxH = 0;
								
								int maxW = 0;
								
								for(int m = 0;m<pHeight;m++) {
									
									for(int n = 0;n<pWidth;n++) {
										
										switch (poolingType) {
										case MAX_POOLING:
											
											if(m == 0 && n == 0) {
												result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
											}else if(result[index][c][i][j] <= x[index][c][i * stride + m][j * stride + n]) {
												result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
												maxH = m;
												maxW = n;
											}

											break;
										case MEAN_POOLING:
											result[index][c][i][j] += x[index][c][i * stride + m][j * stride + n];
											mask[index][c][maskIndex][m][n] = 1.0d / pWidth / pHeight;
											break;
										}
										
									}
									
								}
								
								switch (poolingType) {
								case MAX_POOLING:
									mask[index][c][maskIndex][maxH][maxW] = 1;
									break;
								case MEAN_POOLING:
									result[index][c][i][j] /= pWidth * pHeight;
									break;
								}
								
								maskIndex++;
							}
							
						}
						
					}
					
					return null;
				}
				
			});
			
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return result;
	}
	
	/**
	 * poolingDiff
	 * @param delta
	 * @param mask
	 * @return
	 */
	public static double[][][] poolingDiff(double[][][] delta,double[][][][] mask,double[][][] diff,int pWidth,int pHeight,int stride){
		
		int channel = diff.length;
		
		int oHeight = ((diff[0].length - pHeight ) / stride) + 1;
		
		int oWidth = ((diff[0][0].length - pWidth) / stride) + 1;
		
		for(int c = 0;c<channel;c++) {
			
			int maskIndex = 0;
			
			for(int i = 0;i<oHeight;i++) {
				
				for(int j = 0;j<oWidth;j++) {
					
					for(int m = 0;m<pHeight;m++) {
						
						for(int n = 0;n<pWidth;n++) {
							
							diff[c][i * stride + m][j * stride + n] += delta[c][i][j] * mask[c][maskIndex][m][n];
							
						}
						
					}
					
					maskIndex++;
					
				}
				
			}
			
		}
		
		return diff;
	}
	
	/**
	 * poolingDiff
	 * @param delta
	 * @param mask
	 * @return
	 */
	public static double[][][][] poolingDiff(double[][][][] delta,double[][][][][] mask,double[][][][] diff,int pWidth,int pHeight,int stride){
		
		int number = diff.length;
		
		int channel = diff[0].length;
		
		int oHeight = ((diff[0][0].length - pHeight ) / stride) + 1;
		
		int oWidth = ((diff[0][0][0].length - pWidth) / stride) + 1;
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int b = 0;b<number;b++) {
			
			final int index = b;
			
			workers.add(new Task<Object>(index) {
				
				@Override
			    public Object call() throws Exception {

					for(int c = 0;c<channel;c++) {
						
						int maskIndex = 0;
						
						for(int i = 0;i<oHeight;i++) {
							
							for(int j = 0;j<oWidth;j++) {
								
								for(int m = 0;m<pHeight;m++) {
									
									for(int n = 0;n<pWidth;n++) {
										
										diff[index][c][i * stride + m][j * stride + n] += delta[index][c][i][j] * mask[index][c][maskIndex][m][n];
										
									}
									
								}
								
								maskIndex++;
								
							}
							
						}
						
					}
					
					return null;
				}
			});
			
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return diff;
	}
	
}
