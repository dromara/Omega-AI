package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;

public class Dilation extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 359367116454030645L;

	private int start = 0;
	
	private int end = 0;
	
	private float[][][][] x;
	
	private float[][][][] y;
	
	private int stride = 2;
	
	private int N = 0;
	private int C = 0;
	private int H = 0;
	private int W = 0;
	
	public Dilation(float[][][][] x,float[][][][] y,int N,int C,int H,int W,int stride,int start,int end) {
		this.x = x;
		this.y = y;
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.stride = stride;
		this.start = start;
		this.end = end;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= x.length / 8) {
			
			execute();

		} else {

			int mid = (start + end + 1) >>> 1;
			Dilation left = new Dilation(x, y, N, C, H, W, stride, start, mid - 1);
			Dilation right = new Dilation(x, y, N, C, H, W, stride, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
		}
	}
	
	public void execute() {
		
		int s = stride - 1;
		
		for(int n = start;n<=end;n++) {
			
			for(int c = 0;c<C;c++) {
				
				for(int h = 0;h<H;h++) {

					int hi = h;
					
					if(h > 0) {
						hi = h + h * s;
					}
					
					for(int w = 0;w<W;w++) {
						
						int wi = w;
						
						if(w > 0) {
							wi = w + w * s;
						}
						
						y[n][c][hi][wi] = x[n][c][h][w];
						
					}
					
				}
				
			}
			
		}

	}
	
	public static void dilation(float[][][][] x,float[][][][] y,int stride) {
		
		int N = x.length;
		int C = x[0].length;
		int H = x[0][0].length;
		int W = x[0][0][0].length;
		
		Dilation job = new Dilation(x, y, N, C, H, W, stride, 0, N - 1);
		
		ForkJobEngine.run(job);
		
	}
	
}
