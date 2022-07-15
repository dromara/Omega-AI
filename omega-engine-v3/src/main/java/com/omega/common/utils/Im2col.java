package com.omega.common.utils;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2col extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private float[][][][] x;

	private float[][] colMatrix;

	private int fh, fw, start, end, stride, pad,oh,ow;
	
	public Im2col(float[][][][] x, int fh, int fw, int oh ,int ow,int stride, int pad, float[][] col, int start, int end) {
		// TODO Auto-generated constructor stub
		this.x = x;

		this.colMatrix = col;

		this.start = start;
		this.end = end;
		this.fh = fh;
		this.fw = fw;
		this.stride = stride;
		this.pad = pad;
		
		this.oh = oh;
		this.ow = ow;
	}

	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= x.length / 8) {
			
			toCol();

		} else {

			int mid = (start + end + 1) >>> 1;
			Im2col left = new Im2col(x, fh, fw,oh,ow,stride, pad, colMatrix, start, mid - 1);
			Im2col right = new Im2col(x, fh, fw,oh,ow, stride, pad, colMatrix, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
		}
	}

	private void toCol() {
		
		for (int n = start; n <= end; n++) {
			
			int posX = 0;
			int posY = 0;
			
			for (int i = 0; i < oh; i++) {				
				for (int j = 0; j < ow; j++) {
					int pos = 0;					
					float[] col = colMatrix[n * oh * ow + i * oh + j];
					
					for (int c = 0; c < x[n].length; c++) {
						toCol(x[n][c], col, pos, posX, posY);
						pos += fh * fw;
					}
					posY = j + stride;
				}
				posX = i + stride;
				posY = 0;
			}
		}
	}

	private void toCol(float[][] s, float[] t, int pos, int i, int j) {
		for (int k = i; k < i + fh; k++) {
			System.arraycopy(s[k], j, t, pos, fw);
			pos += fw;
		}
	}
	
	public static void testIm2colInput() {

		ForkJoinPool forkJoinPool = new ForkJoinPool();
		int pad = 0;

		float[][][][] x = new float[][][][] {
			{
				{
					{1.1f,1.2f,1.3f},
					{1.4f,1.5f,1.6f},
					{1.7f,1.8f,1.9f}
				},
				{
					{1.101f,1.11f,1.12f},
					{1.13f,1.14f,1.15f},
					{1.16f,1.17f,1.18f}
				},
				{
					{1.19f,1.201f,1.21f},
					{1.22f,1.23f,1.24f},
					{1.25f,1.26f,1.27f}
				}
			},
			{{{2.1f,2.2f,2.3f},{2.4f,2.5f,2.6f},{2.7f,2.8f,2.9f}},{{2.101f,2.11f,2.12f},{2.13f,2.14f,2.15f},{2.16f,2.17f,2.18f}},{{2.19f,2.201f,2.21f},{2.22f,2.23f,2.24f},{2.25f,2.26f,2.27f}}},
			{{{3.1f,3.2f,3.3f},{3.4f,3.5f,3.6f},{3.7f,3.8f,3.9f}},{{3.101f,3.11f,3.12f},{3.13f,3.14f,3.15f},{3.16f,3.17f,3.18f}},{{3.19f,3.201f,3.21f},{3.22f,3.23f,3.24f},{3.25f,3.26f,3.27f}}},
			{{{4.1f,4.2f,4.3f},{4.4f,4.5f,4.6f},{4.7f,4.8f,4.9f}},{{4.101f,4.11f,4.12f},{4.13f,4.14f,4.15f},{4.16f,4.17f,4.18f}},{{4.19f,4.201f,4.21f},{4.22f,4.23f,4.24f},{4.25f,4.26f,4.27f}}},
			};

		int N = 4;
		int C = 3;
		int H = 3;
		int W = 3;
		
//		float[][][][] x = RandomUtils.gaussianRandom(N, C, H, W, 0.1f);
		
		int stride = 1;

		int fh = 2;
		int fw = 2;

		long start1 = System.nanoTime();
		
		int oh = (H + 2 * pad - fh) / stride + 1;
		int ow = (W + 2 * pad - fw) / stride + 1;
		
		float[][] col = new float[N * oh * ow][fh * fw * C];

		Im2col im2col = new Im2col(x, fh, fw, oh, ow, stride, pad, col, 0, N - 1);
		ForkJoinTask<Void> a = forkJoinPool.submit(im2col);

		a.join();
		
		System.out.println("time1:"+(System.nanoTime() - start1) / 1e6 + "ms.["+col[0][0]+","+col[col.length - 1][col[0].length - 1]+"]");
		
		PrintUtils.printImage(col);
		
		System.out.println("===========================>");
		
//		long start2 = System.nanoTime();
//		
//		float[][] col2 = MatrixOperation.im2col(x, fh, fw, stride);
//		
//		System.out.println("time2:"+(System.nanoTime() - start2) / 1e6 + "ms.["+col2[0][0]+","+col2[col2.length - 1][col2[0].length - 1]+"]");
//
//		PrintUtils.printImage(col2);
//		
//		System.out.println("===========================>");
		
		long start3 = System.nanoTime();
		
		float[][] col3 = Im2col4d.im2col(x, fh, fw, stride);
		
//		float[][] col3 = MatrixOperation.im2col4d(x, fh, fw, stride);
		MatrixOperation.im2col4d(x, fh, fw, stride);
		
		System.out.println("time3:"+(System.nanoTime() - start3) / 1e6 + "ms.["+col3[0][0]+","+col3[col3.length - 1][col3[0].length - 1]+"]");
		
		PrintUtils.printImage(col3);

//		float[][] col4 = MatrixOperation.im2col4d2(x, fh, fw, stride);
//		
//		PrintUtils.printImage(col4);
//		
//		System.out.println("===========================>");
//		
//		float[][] col5 = MatrixUtils.transpose(Im2colUtils.im2col(x, fh, fw, stride));
//		
//		PrintUtils.printImage(col5);
		
	}
	
	public static void main(String[] args) {
		Im2col.testIm2colInput();
	}
	
	
}
