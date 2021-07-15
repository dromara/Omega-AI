package com.omega.common.utils;

/**
 * PrintUtils
 * @author Administrator
 *
 */
public class PrintUtils {
	
	/**
	 * print matrix
	 * @param data
	 */
	public static void printImage(double[] data) {
		
		for(int i = 0;i<data.length;i++) {
			System.out.print(data[i]+" ");
		}
		
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
	
	public static void printImageToOne(double[][][] data) {
		
		for(int c = 0;c<data.length;c++) {
			
			for(int i = 0;i<data[c].length;i++) {
				for(int j = 0;j<data[c][i].length;j++) {
					if(data[c][i][j] == 0.0) {
						System.out.print("0 ");
					}else {
						System.out.print("1 ");
					}
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
	
}
