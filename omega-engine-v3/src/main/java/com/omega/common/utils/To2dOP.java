package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;

public class To2dOP extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1717520538440123479L;

	private float[] x;
	
	private float[][] y;
	
	private int start;
	
	private int end;
	
	private int n;
	
	private int d;
	
	public To2dOP(float[] x,float[][] y,int start,int end) {
		// TODO Auto-generated constructor stub
		this.x = x;
		this.y = y;
		this.start = start;
		this.end = end;
		this.n = y.length;
		this.d = y[0].length;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= x.length / 8) {
			
			execute();

		} else {

			int mid = (start + end + 1) >>> 1;
			To2dOP left = new To2dOP(x, y, start, mid - 1);
			To2dOP right = new To2dOP(x, y, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
		}
	}

	
	public void execute() {
		
		for(int i = start;i<=end;i++) {
			System.arraycopy(x, i * d, y[i], 0, d);
		}
		
	}
	
	public static float[][] to2dArray(float[] x,int n,int d){
		float[][] result = new float[n][d];
		
		To2dOP job = new To2dOP(x, result, 0, n - 1);
		
		ForkJobEngine.run(job);
		
		return result;
	}
	
}
