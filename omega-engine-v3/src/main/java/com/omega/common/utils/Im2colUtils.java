package com.omega.common.utils;

import java.util.Vector;

import com.omega.common.task.ForkJobEngine;
import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.engine.gpu.data.CacheDataSet;

/**
 * Im2colUtils
 * @author Administrator
 *
 */
public class Im2colUtils {
	
	private static final int threadNum = 8;
	
	public static float[][] im2col(float[][][][] x,int kh,int kw,int stride){
		
		int N = x.length;

		int C = x[0].length;

		int oHeight = ((x[0][0].length - kh) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
//		long start = System.nanoTime();
		
		float[][] col = new float[N * oHeight * oWidth][kh * kw * C];

//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.1111");
		
//		long start2 = System.nanoTime();
		
		Im2col task = new Im2col(x, kh, kw, oHeight, oWidth, stride, 0, col, 0, N - 1);
		
		ForkJobEngine.run(task);
		
//		System.out.println((System.nanoTime() - start2) / 1e6 + "ms.22222");
		
		return col;
	}
	
	public static float[][] im2col(float[][][][] x,int kh,int kw,int stride,CacheDataSet cahce){
		
		int N = x.length;

		int C = x[0].length;

		int oHeight = ((x[0][0].length - kh) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
//		long start = System.nanoTime();
		
		float[][] col = cahce.getDim2dSet().get(0);

//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.1111");
		
//		long start2 = System.nanoTime();
		
		Im2col task = new Im2col(x, kh, kw, oHeight, oWidth, stride, 0, col, 0, N - 1);
		
		ForkJobEngine.run(task);
		
//		System.out.println((System.nanoTime() - start2) / 1e6 + "ms.22222");
		
		return col;
	}
	
	public static float[][] im2colKernel(float[][][][] x){
		
		int ko = x.length;
		int kc = x[0].length;
		int kh = x[0][0].length;
		int kw = x[0][0][0].length;
		
		/**
		 * kernel im2col
		 */
		float[][] col = new float[ko][kh * kw * kc];

		Im2col task = new Im2col(x, kh , kw, 1, 1, 0, 0, col, 0, ko - 1);
		
		ForkJobEngine.run(task);

		return col;
	}
	
	public static float[][] kernalTo2d(float[][][][] x){
		int N = x.length;
		int C = x[0].length;
		int H = x[0][0].length;
		int W = x[0][0][0].length;
		
		float[][] y = new float[N * H * W][C];
		
		KernalTo2d job = new KernalTo2d(x, y, 0, N * H * W - 1);
		
		ForkJobEngine.run(job);
		
		return y;
	}
	
	/**
	 * 
	 * @param x
	 * @param isT (true : [N * H * W][C]  , false : [C * H * W][N])
	 * @return
	 */
	public static float[] kernalToVector(float[][][][] x,boolean isT){
		int N = x.length;
		int C = x[0].length;
		int H = x[0][0].length;
		int W = x[0][0][0].length;
		
		float[] y = new float[N * H * W * C];
		KernalToVector job = null;
		if(isT) {
			job = new KernalToVector(x, y, 0, N * H * W - 1, isT);	
		}else {
			job = new KernalToVector(x, y, 0, C * H * W - 1, isT);
		}
		ForkJobEngine.run(job);
			
		return y;
	}
	
	public static float[][] to2d(float[][][][] x){

		int N = x.length;
		int C = x[0].length;
		int H = x[0][0].length;
		int W = x[0][0][0].length;
		
		float[][] col = new float[N * H * W][C];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<N;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int h = 0;h<H;h++) {
						for(int w = 0;w<W;w++){
							for(int c = 0;c<C;c++) {
								col[index * H * W + h * W + w][c] = x[index][c][h][w];
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);
		
		return col;
	}
	
	public static float[][][][] to4d(float[][] x,int N,int C,int H,int W){
		
		float[][][][] result = new float[N][C][H][W];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<N;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								result[index][c][h][w] = x[c * H * W + h * W + w][index];
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
	
	public static float[][][][] to4d(float[] x,int N,int C,int H,int W){
		
		float[][][][] result = new float[N][C][H][W];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<N;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								result[index][c][h][w] = x[index * C * H * W + c * H * W + h * W + w];
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
	
	public static void to4d(float[] x,float[][][][] y,int N,int C,int H,int W){
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<N;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								y[index][c][h][w] = x[index * C * H * W + c * H * W + h * W + w];
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(threadNum).dispatchTask(workers);

	}
	
}
