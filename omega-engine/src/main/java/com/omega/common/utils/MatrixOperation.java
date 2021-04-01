package com.omega.common.utils;

import java.math.BigDecimal;
import java.util.Random;

import com.omega.engine.active.Relu;
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
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = Math.exp(x[i]);
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
		double[] temp = MatrixOperation.zero(x.length);
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] + b;
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
		double[] temp = MatrixOperation.zero(x.length);
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
	public static double[][] subtraction(double[][] x,double b) {
		if(x == null) {
			new RuntimeException("matrix is null.");
		}
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] - b;
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
		double[] temp = MatrixOperation.zero(x.length);
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] * b;
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
		double[] temp = MatrixOperation.zero(x.length);
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] / b;
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
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] + b[i];
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
		double[] temp = MatrixOperation.zero(x.length);
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
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] - b[i];
			String str=new BigDecimal(b[i]).toString();
			System.out.println(x[i] + " - " + str);
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
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] * b[i];
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
		double[] temp = MatrixOperation.zero(x.length);
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i] / b[i];
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
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
		
		double[][][] temp = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
		
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
	public static double[][][] add(double[][][] x,double[][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		
		if(x.length != b.length) {
			new RuntimeException("x size must equals b.");
		}
		
		double[][][] temp = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
		
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
	public static double[][] subtraction(double[][] x,double[][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j] - b[i][j];
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
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
	public static double[][][] multiplication(double[][][] x,double[][][] b) {
		if(x == null || b == null) {
			new RuntimeException("matrix is null.");
		}
		if(x.length != b.length || x[0].length != b[0].length || x[0][0].length != b[0][0].length) {
			new RuntimeException("x size must equals b.");
		}
		double[][][] temp = MatrixOperation.zero(x.length,x[0].length,x[0][0].length);
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
		double[][] temp = MatrixOperation.zero(x.length,x[0].length);
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[0].length;j++) {
				temp[i][j] = x[i][j] / b[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: clone
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] clone(double[] x) {
		double[] temp = new double[x.length];
		for(int i = 0;i<x.length;i++) {
			temp[i] = x[i];
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: clone
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] clone(double[][] x) {
		double[][] temp = new double[x.length][x[0].length];
		for(int i = 0;i<x.length;i++) {
			for(int j = 0;j<x[i].length;j++) {
				temp[i][j] = x[i][j];
			}
		}
		return temp;
	}
	
	/**
	 * 
	 * @Title: clone
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] clone(double[][][] x) {
		double[][][] temp = new double[x.length][x[0].length][x[0][0].length];
		for(int c = 0;c<x.length;c++) {
			for(int i = 0;i<x[c].length;i++) {
				for(int j = 0;j<x[c][i].length;j++) {
					temp[c][i][j] = x[c][i][j];
				}
			}
		}
		return temp;
	}
	
	/**
	 * transform
	 * @param x  c * h * w
	 * @return
	 */
	public static double[] transform(double[][][] x) {
		
		double[] result = new double[x.length * x[0].length * x[0][0].length];
		
		for(int c = 0;c<x.length;c++) {
			
			for(int i = 0;i<x[c].length;i++) {
				
				for(int j = 0;j<x[c][i].length;j++) {
					
					result[c*i*j + i*j +j] = x[c][i][j];
					
				}
			}
			
		}
		
		return result;
	}
	

	/**
	 * transform
	 * @param x
	 * @return
	 */
	public static double[][][] transform(double[] x,int c,int h,int w) {
		
		double[][][] y = new double[c][h][w];
		
		for(int ci = 0;ci<c;ci++) {
			for(int hi = 0;hi<h;hi++) {
				for(int wi = 0;wi<w;wi++) {
					y[ci][hi][wi] = x[ci*hi*wi + hi*wi + wi];
				}
			}
		}
		
		return y;
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
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] zero(int size) {
		return new double[size];
	}
	
	/**
	 * 
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][] zero(int x,int y) {
		return new double[x][y];
	}
	
	/**
	 * 
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][] zero(int x,int y,int z) {
		return new double[x][y][z];
	}
	
	/**
	 * 
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] zero(int x,int y,int z,int n) {
		return new double[x][y][z][n];
	}
	
	/**
	 * 
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][][] zero(int x,int y,int z,int n,int m) {
		return new double[x][y][z][n][m];
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
	 * rotate90
	 * @param matrix
	 * @return
	 */
	public static double[][] rotate90(double [][]matrix){
		double [][]temp=new double[matrix[0].length][matrix.length];
		int dst=matrix.length-1;
		for(int i=0;i<matrix.length;i++,dst--){
			for(int j=0;j<matrix[0].length;j++){
				temp[j][dst]=matrix[i][j];
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
	 * convnVail
	 * @param x  c * h * w
	 * @param k  c * kn * kh * kw
	 * @param stride
	 * @return
	 * 
	 * o = (W - K) / S + 1
	 */
	public static double[][][] convnVail(double[][][][] k,double[][][] x,int stride){
		
		int channel = k.length;
		
		int kNum = k[0].length;
		
		int oHeight = ((x[0].length - k[0][0].length ) / stride) + 1;
		
		int oWidth = ((x[0][0].length - k[0][0][0].length) / stride) + 1;

		double[][][] diff = new double[channel][oHeight][oWidth];

		for(int c = 0;c<channel;c++) {
			
			for(int kn = 0;kn<kNum;kn++) {
				
				for(int i = 0;i<oHeight;i++) {
					
					for(int j = 0;j<oWidth;j++) {
						
						for(int m = 0;m<k[c][kn].length;m++) {
								
							for(int n = 0;n<k[c][kn][m].length;n++) {
									
								diff[c][i][j] += x[kn][i * stride + m][j * stride + n] * k[c][kn][m][n];
	
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
								
								result[c][kn][i][j] = x[c][i * stride + m][j * stride + n] * d[kn][m][n];

							}
								
						}
							
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
								}else if(result[c][i][j] < x[c][i * stride + m][j * stride + n]) {
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
	 * 高斯随机数
	 * @param x
	 * @return
	 */
	public static double[] gaussianRandom(int x,double ratio){
		double[] temp = new double[x];
		Random r = new Random();
		for(int i = 0;i<x;i++) {
			temp[i] = r.nextGaussian() / Math.sqrt(x);
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
		Random r = new Random();
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = r.nextGaussian() / Math.sqrt(x);
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
		Random r = new Random();
		for(int k = 0;k<c;k++) {
			for(int l = 0;l<n;l++) {
				for(int i = 0;i<x;i++) {
					for(int j = 0;j<y;j++) {
						temp[k][l][i][j] = r.nextGaussian() / Math.sqrt(x);
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
		Random r = new Random();
		for(int i = 0;i<x;i++) {
			temp[i] = r.nextGaussian() / Math.sqrt(x);
		}
		return temp;
	}
	
	/**
	 * xavier随机数
	 * @param x
	 * @return
	 */
	public static double[][] xavierRandom(int x,int y,double ratio){
		double[][] temp = new double[x][y];
		Random r = new Random();
		for(int i = 0;i<x;i++) {
			for(int j = 0;j<y;j++) {
				temp[i][j] = r.nextGaussian() / Math.sqrt(x);
			}
		}
		return temp;
	}
	 
	/**
	 * 
	 * @Title: zero
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[] clear(double[] x) {
		if(x != null) {
			x = MatrixOperation.zero(x.length);
		}
		return x;
	}
	
	
	/**
	 * print matrix
	 * @param data
	 */
	public static void printImage(double[][] data) {
		
		for(int i = 0;i<data.length;i++) {
			for(int j = 0;j<data[i].length;j++) {
				System.out.print(data[i][j]+" ");
			}
			System.out.println("");
		}
		
	}
	
	public static void printImage(double[][][] data) {
		
		for(int c = 0;c<data.length;c++) {
			
			for(int i = 0;i<data[c].length;i++) {
				for(int j = 0;j<data[c][i].length;j++) {
					System.out.print(data[c][i][j]+" ");
				}
				System.out.println("");
			}
			
			System.out.println("-------------------------");
			
		}
		
	}
	
	public static void printImage(double[][][][] data) {
		
		for(int m = 0;m<data.length;m++) {
		
			for(int c = 0;c<data[m].length;c++) {
				
				for(int i = 0;i<data[m][c].length;i++) {
					for(int j = 0;j<data[m][c][i].length;j++) {
						System.out.print(data[m][c][i][j]+" ");
					}
					System.out.println("");
				}
				
				System.out.println("-------------------------");
				
			}

			System.out.println("=============================");
			
		}
	}
	
	public static void main(String[] args) {
		
//		double[][] image = {
//                        {0,2,4,2},
//                        {1,2,6,4},
//                        {9,0,2,5},
//                        {6,3,2,6}
//                        };
//		
//		double[][] kernel = {
//				{1,1,1},
//				{2,0,3},
//				{1,0,1}
//		};
//		
//		double[][] result = MatrixOperation.convn(image, kernel, 1, ConvnType.SAME);
//		
//		MatrixOperation.printImage(result);
//		
//		System.out.println();
//		
//		MatrixOperation.printImage(MatrixOperation.rotate180(kernel));

		
//		double[] test = {0.0012059797803661777,1.9300205425211106E-4,0.0012743763716464667,0.2089237171602954,2.0094831831138366E-6,0.16176585639610447,1.845016637757435E-6,0.006165115979960441,0.6197898450559284,6.78252701625735E-4};
//		
//		System.out.println(MatrixOperation.max(test));
//		System.out.println(MatrixOperation.maxIndex(test));

//		double[][] image = {
//			{7,2,4,2},
//			{1,2,6,4},
//			{9,0,2,5},
//			{6,3,2,6}
//		};
//		
//		double[][] pInput = MatrixOperation.zeroPadding(image, 1);
//		
//		MatrixOperation.printImage(image);
//		
//		System.out.println();
//		
//		MatrixOperation.printImage(pInput);
//		
//		double[][] kernel = {
//				{1,1,1},
//				{2,0,3},
//				{1,0,1}
//		};
//		
//		System.out.println();
//		
//		MatrixOperation.printImage(kernel);
//		
//		double[][] temp = MatrixOperation.convnVail(pInput, kernel, 2);
//
//		System.out.println();
//		
//		MatrixOperation.printImage(temp);
		
		
		double[][][] image = {
				{
					{7,2,4,2},
					{1,2,6,4},
					{9,0,2,5},
					{6,3,2,6}
				},
				{
					{2,2,4,2},
					{3,0,0,4},
					{-9,-1,0,5},
					{-1,-3,0,6}
				},
				{
					{1,1,1,4},
					{3,1,0,1},
					{4,1,0,1},
					{0,1,1,6}
				}
				};
			
			double[][][] pInput = MatrixOperation.zeroPadding(image, 1);
			
			MatrixOperation.printImage(image);
			
			System.out.println();
			
			MatrixOperation.printImage(pInput);
			
			double[][][][] kernel = {
					{
						{
							{3,-1,1},
							{2,0,3},
							{1,0,1}
						},
						{
							{1,0,3},
							{0,1,0},
							{1,0,1}
						},
						{
							{3,3,2},
							{-2,0,3},
							{1,1,2}
						},
						{
							{3,3,2},
							{-2,0,3},
							{1,1,3}
						}
					},
					{
						{
							{2,-1,1},
							{2,0,3},
							{1,0,1}
						},
						{
							{-1,2,1},
							{0,1,0},
							{1,0,1}
						},
						{
							{2,-3,2},
							{2,0,-3},
							{1,1,2}
						},
						{
							{3,3,2},
							{-2,1,3},
							{1,1,2}
						}
					},
					{
						{
							{0,-1,1},
							{2,0,3},
							{-1,0,1}
						},
						{
							{-1,0,0},
							{0,1,0},
							{1,0,-1}
						},
						{
							{1,-3,2},
							{2,0,3},
							{-1,1,2}
						},
						{
							{3,2,2},
							{-2,0,3},
							{1,1,2}
						}
					}
			};
			
			double[] bias = {0.1,0.2,0.3,0.5};
			
			double[][][] temp = MatrixOperation.convnVail(pInput, kernel, 1);
			
			System.out.println("conv");

			MatrixOperation.printImage(temp);
			
			double[][][] tempB = MatrixOperation.add(temp, bias);
			
			System.out.println("bias:");

			MatrixOperation.printImage(tempB);
			
			Relu relu = new Relu();
			
			double[][][] aTemp = relu.activeTemp(tempB);
			
			System.out.println("active:");

			MatrixOperation.printImage(aTemp);
			
			double[][][][] mask = new double[4][9][2][2];
			
			double[][][] poolingTemp = MatrixOperation.poolingAndMask(aTemp, mask, 2, 2, 2, PoolingType.MAX_POOLING);
			
			System.out.println("pooling:");

			MatrixOperation.printImage(poolingTemp);
			
			System.out.println("pooling-mask:");

			MatrixOperation.printImage(mask);
			
			double[][][] delta = {
					{
						{0.1,0.3,0.5},
						{0.2,-0.5,0.7},
						{-0.4,0.8,0.9}
					},
					{
						{0.1,0.1,0.2},
						{-0.1,0.5,-0.9},
						{0.2,0.5,0.3}
					},
					{
						{0.1,0.1,0.1},
						{0.2,0.2,0.2},
						{0.5,0.5,0.4}
					},
					{
						{0.1,0.1,0.1},
						{0.2,0.3,0.3},
						{0.4,0.1,0.1}
					}
			};
			
			double[][][] diff = new double[4][4][4];
			
			diff = MatrixOperation.poolingDiff(delta, mask, diff, 2, 2, 2);
			
			System.out.println("pooling-diff:");

			MatrixOperation.printImage(diff);
			
			double[][][] adiffTemp = relu.diffTemp(tempB);
			
			System.out.println("adiffTemp:");
			
			/**
			 * adiffTemp
			 */
			MatrixOperation.printImage(adiffTemp);
			
			double[][][] deltaTemp = MatrixOperation.multiplication(diff, adiffTemp);
			
			System.out.println("deltaTemp:");
			
			/**
			 * deltaTemp
			 */
			MatrixOperation.printImage(deltaTemp);
			
			System.out.println("deltaW:");
			
			/**
			 * 计算deltaW
			 */
			MatrixOperation.printImage(MatrixOperation.convnVail(pInput, deltaTemp, 1));
			
			/**
			 * 梯度添加zeroPadding使得size与卷积输入一致
			 */
			double[][][] deltaP = MatrixOperation.zeroPadding(deltaTemp, 1);
			
			/**
			 * 计算当前层梯度
			 */
			double[][][] diffP = MatrixOperation.convnVail(kernel, deltaP, 1);
			
			System.out.println("diff:");
			
			MatrixOperation.printImage(diffP);
			
//			System.out.println("diff-nozero:");
//			
//			/**
//			 * 去除输入层zeroPadding
//			 */
//			MatrixOperation.printImage(MatrixOperation.dislodgeZeroPadding(diffP, 1));
			
	}
	
}
