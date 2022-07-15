package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class OP2dTo4d extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6548333970817152115L;
	
	private float[][] x;
	
	private float[][][][] y;
	
	private int start;
	
	private int end;
	
	private int N;
	private int C;
	private int H;
	private int W;
	
	public OP2dTo4d(float[][] x,float[][][][] y,int start,int end) {
		this.x = x;
		this.y = y;
		this.N = y.length;
		this.C = y[0].length;
		this.H = y[0][0].length;
		this.W = y[0][0][0].length;
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
			OP2dTo4d left = new OP2dTo4d(x, y, start, mid - 1);
			OP2dTo4d right = new OP2dTo4d(x, y, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
		}
		
	}
	
	private void execute() {
		
		for(int i = start; i <= end; i++) {

			int n = i / H / W;
			
			int h = (i - (n * H * W)) / W;
			
			int w = (i - (n * H * W)) % W;
			
			for(int c = 0;c<C;c++){
				
				y[n][c][h][w] = x[i][c];
				
			}
			
		}
		
	}
	
}
