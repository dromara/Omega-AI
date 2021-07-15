package com.omega.common.utils;

/**
 * MatrixUtils
 * @author Administrator
 *
 */
public class MatrixUtils {
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
	public static double[] one(int size) {
		double[] temp = new double[size];
		for(int i = 0;i<size;i++) {
			temp[i] = 1.0d;
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
	public static double[][][] createMatrix(int channel,int heigth,int width,double value) {
		double[][][] temp = new double[channel][heigth][width];
		for(int c = 0;c<channel;c++) {
			for(int h = 0;h<heigth;h++) {
				for(int w = 0;w<width;w++) {
					temp[c][h][w] = value;
				}
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
	public static double[][][] createMatrixByIndex(int channel,int heigth,int width) {
		double[][][] temp = new double[channel][heigth][width];
		
		for(int c = 0;c<channel;c++) {
			double index = 1;
			for(int h = 0;h<heigth;h++) {
				for(int w = 0;w<width;w++) {
					temp[c][h][w] = index;
					index++;
				}
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
	public static double[][][][] createMatrixByIndex(int number,int channel,int heigth,int width) {
		double[][][][] temp = new double[number][channel][heigth][width];
		for(int n = 0;n<number;n++) {
			for(int c = 0;c<channel;c++) {
				double index = 1;
				for(int h = 0;h<heigth;h++) {
					for(int w = 0;w<width;w++) {
						temp[n][c][h][w] = index;
						index++;
					}
				}
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
	 * @Title: val for matrix
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] val(int x,int y,int z,int n, double val) {
		double[][][][] tmp = new double[x][y][z][n];
		
		for(int c = 0;c<tmp.length;c++) {
			for(int i = 0;i<tmp[c].length;i++) {
				for(int j = 0;j<tmp[c][i].length;j++) {
					for(int l = 0;l<tmp[c][i][j].length;l++) {
						tmp[c][i][j][l] = val;
					}
				}
			}
		}
		
		return tmp;
	}
	
	/**
	 * 
	 * @Title: val for matrix
	 *
	 * @param size
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 *
	 * @throws
	 */
	public static double[][][][] val(int x,int y,int z,int n, double p,double val) {
		double[][][][] tmp = new double[x][y][z][n];
		
		for(int c = 0;c<tmp.length;c++) {
			for(int i = 0;i<tmp[c].length;i++) {
				for(int j = 0;j<tmp[c][i].length;j++) {
					for(int l = 0;l<tmp[c][i][j].length;l++) {
						if(RandomUtils.getInstance().nextFloat() > p) {
							tmp[c][i][j][l] = val;
						}
					}
				}
			}
		}
		
		return tmp;
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
			x = MatrixUtils.zero(x.length);
		}
		return x;
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
	public static double[][] clear(double[][] x) {
		if(x != null) {
			x = MatrixUtils.zero(x.length,x[0].length);
		}
		return x;
	}
	
	/**
	 * size
	 * @param data
	 * @return
	 */
	public static int[] size(double[][][][] data) {
		System.out.println("["+data.length+","+data[0].length+","+data[0][0].length+","+data[0][0][0].length+"]");
		return new int[]{data.length,data[0].length,data[0][0].length,data[0][0][0].length};
	}

	/**
	 * size
	 * @param data
	 * @return
	 */
	public static int[] size(double[][] data) {
		System.out.println("["+data.length+","+data[0].length);
		return new int[]{data.length,data[0].length};
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
	 * @index ci * h * w + hi * w + wi
	 * @return
	 */
	public static double[] transform(double[][][] x) {
		
		double[] result = new double[x.length * x[0].length * x[0][0].length];
		
		for(int c = 0;c<x.length;c++) {
			
			for(int h = 0;h<x[c].length;h++) {
				
				for(int w = 0;w<x[c][h].length;w++) {
					result[c*x[c].length*x[c][h].length + h*x[c][h].length + w] = x[c][h][w];
				}
			}
			
		}
		
		return result;
	}
	
	/**
	 * 
	 * @param x
	 * @index ni * c * h * w + ci * h * w + hi * w + wi
	 * @return
	 */
	public static double[] transform(double[][][][] x){
		
		double[] y = new double[x.length * x[0].length * x[0][0].length * x[0][0][0].length];
		
		for(int n = 0;n<x.length;n++) {
			for(int c = 0;c<x[n].length;c++) {
				for(int h = 0;h<x[n][c].length;h++) {
					for(int w = 0;w<x[n][c][h].length;w++) {
						y[n*x[n].length*x[n][c].length*x[n][c][h].length + c*x[n][c].length*x[n][c][h].length + h*x[n][c][h].length + w] = x[n][c][h][w];
					}
				}
				
			}
		}
		return y;
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
					y[ci][hi][wi] = x[ci * h * w + hi * w + wi];
				}
			}
		}
		
		return y;
	}
	
	/**
	 * transform
	 * @param x
	 * @index ni * c * h * w + ci * h * w + hi * w + wi
	 * @return
	 */
	public static double[][][][] transform(double[] x,int n,int c,int h,int w) {
		
		double[][][][] y = new double[n][c][h][w];
		for(int ni = 0;ni<n;ni++) {
			for(int ci = 0;ci<c;ci++) {
				for(int hi = 0;hi<h;hi++) {
					for(int wi = 0;wi<w;wi++) {
						y[ni][ci][hi][wi] = x[ni * c * h * w + ci * h * w + hi * w + wi];
					}
				}
			}
		}
		
		return y;
	}
	
	/**
	 * transform
	 * @param x
	 * @return
	 */
	public static double[][][][] transform(double[][] x,int n,int c,int h,int w) {
		
		double[][][][] y = new double[n][c][h][w];
		
		for(int ni = 0;ni<n;ni++) {
			for(int ci = 0;ci<c;ci++) {
				for(int hi = 0;hi<h;hi++) {
					for(int wi = 0;wi<w;wi++) {
						y[ni][ci][hi][wi] = x[ni][ci * h * w + hi * w + wi];
					}
				}
			}
		}

		return y;
	}
	
	/**
	 * transform
	 * @param x  c * h * w
	 * @return
	 */
	public static double[][][] transform(double[][] x,int oh,int ow) {
		
		int channel = x[0].length;
		
		double[][][] result = new double[channel][oh][ow];
		
		for(int c = 0;c<channel;c++) {
			int index = 0;
			for(int h = 0;h<oh;h++) {
				for(int w = 0;w<ow;w++) {
					result[c][h][w] = x[index][c];
					index++;
				}
			}
		}
		
		return result;
	}
	
	/**
	 * transform
	 * @param x
	 * @return
	 * @remark 
	 * 数据长度必须对等
	 */
	public static double[][][][] transform(double[][][][] x,int n,int c,int h,int w) {
		
		double[] temp = MatrixUtils.transform(x);
		
		return MatrixUtils.transform(temp, n, c, h, w);
	}
	
}
