package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;

public class Im2col4d extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5122995462148301836L;

	private int start = 0;
	
	private int end = 0;
	
	private float[][][][] x;
	
	private float[][] y;
	
	private int kh;
	private int kw;
	private int stride;
	
	
	public Im2col4d(float[][][][] data,float[][] col,int kh,int kw,int stride,int start,int end) {
		this.x = data;
		this.y = col;
		this.start = start;
		this.end = end;
		this.kh = kh;
		this.kw = kw;
		this.stride = stride;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= x.length / 8) {
			
			col();

		} else {

			int mid = (start + end + 1) >>> 1;
			Im2col4d left = new Im2col4d(x, y, kh, kw, stride, start, mid - 1);
			Im2col4d right = new Im2col4d(x, y, kh, kw, stride, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}
	
	private void col() {
		
//		int N = x.length;
		
		int oHeight = ((x[0][0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
		int ow = x[0].length * kh * kw;
		
//		int oh = N * oHeight * oWidth;
		
		int kSize = kh * kw;
		
		for (int i = start; i <= end; i++) {
			
			int n = i / oHeight / oWidth;
			
			int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
			
			int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
			
			for(int j = 0;j<ow;j++) {
				
				int c = j / kSize;
				
				int xSize = j - (c * kSize);
				
				int xh = startH + xSize / kw;
				
				int xw = startW + xSize % kw;
				
				y[i][j] = x[n][c][xh][xw];

			}
			
		}
		
	}
	
	public static float[][] im2col(float[][][][] x,int kh,int kw,int stride){
		
		int N = x.length;
		
		int oHeight = ((x[0][0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
		int ow = x[0].length * kh * kw;
		
		int oh = N * oHeight * oWidth;
		
		long start3 = System.nanoTime();
		    	
		float[][] result = new float[oh][ow];
		
		System.out.println((System.nanoTime() - start3) / 1e6 + "ms=============");
		
		Im2col4d job = new Im2col4d(x, result, kh, kw, stride, 0, oh - 1);
	
		ForkJobEngine.run(job);
		
		return result;
	}
	
	public static float[][] im2colV2(float[][][][] x,float[][] result,int kh,int kw,int stride){
		
		int N = x.length;
		
		int oHeight = ((x[0][0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
//		int ow = x[0].length * kh * kw;
		
		int oh = N * oHeight * oWidth;
		
		Im2col4d job = new Im2col4d(x, result, kh, kw, stride, 0, oh - 1);
	
		ForkJobEngine.run(job);
		
		return result;
	}
	
}
