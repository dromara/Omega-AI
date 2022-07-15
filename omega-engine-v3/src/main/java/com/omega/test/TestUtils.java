package com.omega.test;

import java.util.concurrent.ForkJoinPool;

import com.omega.common.utils.Dilation;
import com.omega.common.utils.Im2colForWeight;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;

public class TestUtils {
	
	
	public static float[][][][] getX(){
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
		return x;
	}
	
	public static float[][] testIm2colInput() {

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

		int stride = 1;

		int kh = 2;
		int kw = 2;
		
		float[][] v1 = Im2colUtils.to2d(x);
		
		PrintUtils.printImage(v1);
	
		return v1;
	}
	
	public static float[][] testKernal() {
		
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		int pad = 0;

		float[][][][] k = new float[][][][] {
			{
				{
					{1.01f,1.02f},
					{1.03f,1.04f}
				},
				{
					{1.101f,1.12f},
					{1.13f,1.14f}
				},
				{
					{1.21f,1.22f},
					{1.23f,1.24f}
				}
			},
			{{{2.01f,2.02f},{2.03f,2.04f}},{{2.101f,2.12f},{2.13f,2.14f}},{{2.21f,2.22f},{2.23f,2.24f}}},
			{{{3.01f,3.02f},{3.03f,3.04f}},{{3.101f,3.12f},{3.13f,3.14f}},{{3.21f,3.22f},{3.23f,3.24f}}}
		};

		int KO = 3;
		int KC = 3;
		int KH = 2;
		int KW = 2;

		int stride = 1;
		
		
//		System.out.println("-----------------------------");
//		
//		PrintUtils.printImage(Im2colUtils.to2d(k));
		
		PrintUtils.printImage(MatrixUtils.transpose(Im2colUtils.im2colKernel(k)));
		
		float[] ka = MatrixUtils.transform(Im2colUtils.kernalTo2d(k));

		System.out.println("=============================");
		
		PrintUtils.printImage(ka);
		System.out.println("");
		System.out.println("------------------------------");
		PrintUtils.printImage(Im2colUtils.kernalToVector(k, false));
		
		
		return null;
	}
	
	/**
	 * im2col2
	 * @param x
	 * @param kh
	 * @param kw
	 * @param stride
	 * @return
	 */
	public static float[][] im2col4d(float[][][][] x,int kh,int kw,int stride){
		
		int N = x.length;
		
		int oHeight = ((x[0][0].length - kh ) / stride) + 1;
		
		int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
		
		int ow = x[0].length * kh * kw;
		
		int oh = N * oHeight * oWidth;
		
		int kSize = kh * kw;
		
		float[][] result = new float[oh][ow];
		
		for(int i = 0;i<oh;i++) {

			int n = i / oHeight / oWidth;
			
			int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
			
			int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
			
			for(int j = 0;j<ow;j++) {
				
				int c = j / kSize;
				
				int xSize = j - (c * kSize);
				
				int xh = startH + xSize / kw;
				
				int xw = startW + xSize % kw;
				
				result[i][j] = x[n][c][xh][xw];

			}
			
		}
		
		return result;
	}
	
	public static void test() {
		
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
		
		float[][][][] k = new float[][][][] {
			{
				{
					{1.01f,1.02f},
					{1.03f,1.04f}
				},
				{
					{1.101f,1.12f},
					{1.13f,1.14f}
				},
				{
					{1.21f,1.22f},
					{1.23f,1.24f}
				}
			},
			{{{2.01f,2.02f},{2.03f,2.04f}},{{2.101f,2.12f},{2.13f,2.14f}},{{2.21f,2.22f},{2.23f,2.24f}}},
			{{{3.01f,3.02f},{3.03f,3.04f}},{{3.101f,3.12f},{3.13f,3.14f}},{{3.21f,3.22f},{3.23f,3.24f}}}
		};

		int kc = k[0].length;
		int kh = k[0][0].length;
		int kw = k[0][0][0].length;
		
		int N = x.length;
		
		int stride = 1;
		
		int h = x[0][0].length;
		
		int w = x[0][0][0].length;
		
		int diffPadding = ((4 - 1) * stride + kh - 3) / 2;
		
		System.out.println(diffPadding);
		
		float[][][][] deltaP = MatrixOperation.zeroPadding(x, diffPadding);
		

		int oHeight = ((deltaP[0][0].length - kh) / stride) + 1;
		
		int oWidth = ((deltaP[0][0][0].length - kw) / stride) + 1;
		
		
		/**
		 * input im2col
		 */

		float[][] input2d = Im2colUtils.im2col(deltaP, kh, kw, 1);
		
		float[] input1d = Im2colToVector.im2col(deltaP, kh, kw, 1);
		
		float[] xa = MatrixUtils.transform(input2d);
		
		System.out.println(JsonUtils.toJson(xa));
		
		System.out.println(JsonUtils.toJson(input1d));
//		
//		/**
//		 * kernel im2col
//		 */
//		float[][][][] kernel180 = MatrixOperation.rotate180V2(k);
//		
//		float[][] kt = Im2colUtils.to2d(kernel180);
//		
//		float[] r = new float[N * kc * oHeight * oWidth];
//		
//		GPUOP.getInstance().multiplyFloat(input2d.length, kt.length, kt[0].length, MatrixUtils.transform(input2d), MatrixUtils.transform(kt), r);
//
//		float[][][][] tmp = MatrixUtils.col2img(r, N, kc, oHeight, oWidth);
//		
//		PrintUtils.printImage(tmp);
		
	}
	
	public static void testWeight() {
		
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
		int kh = 2;
		int kw = 2;
		
		int oHeight = ((3 - kh ) / 1) + 1;
		
		int oWidth = ((3 - kw) / 1) + 1;
		
		int xm = C * oHeight * oWidth;
		int xn = N * kh * kw;

		int pLength = xm * xn;
		
		float[] y = new float[pLength];
		
		Im2colForWeight.im2col(x, y, 2, 2, 1);
		
//		PrintUtils.printImage(y);
		
		for(int i = 0;i<xm;i++) {
			
			System.out.println("");
			
			for(int j = 0;j<xn;j++) {
				
				System.out.print(y[i * xn + j]+" ");
				
			}
			
		}
		
	}
	
	public static void meanTest() {
		
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
		
		float[] y1 =  MatrixOperation.mean(x, 1);
		
		float[] y2 =  MatrixOperation.meanO(x, 1);
		
		System.out.println(JsonUtils.toJson(y1));
		System.out.println(JsonUtils.toJson(y2));
		
	}
	
	public static void dilationTest() {
	
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
		
		float[][][][] y = new float[4][3][7][7];
		
		Dilation.dilation(x, y, 3);
		
		PrintUtils.printImage(y);
		
	}
	
	public static void main(String[] args) {
		
//		TestUtils.testIm2colInput();
//		TestUtils.testKernal();

//		float[][] x2 = TestUtils.testKernal();
		
//		float[][][][] x = TestUtils.getX();
//
//		float[] v1 = MatrixUtils.transform(x);
//		
//		float[][][][] v2 = MatrixUtils.transform(v1, 4, 3, 3, 3);
//		
//		System.out.println("-----------------x-------------------------");
//		PrintUtils.printImage(x);
//		System.out.println("-----------------v1-------------------------");
//		PrintUtils.printImage(v1);
//		System.out.println("");
//		System.out.println("-----------------v2-------------------------");
//		PrintUtils.printImage(v2);
		
//		test();
		
//		int N = 128;
//		int W = 5000;
//		int OW = 5000;
//		
//		float[][] x = RandomUtils.x2Random(N, W);
//		
//		float[][] d = RandomUtils.x2Random(N, OW);
//		
////		float[][] x = new float[][] {
////			{1.0f,2.0f,3.0f,4.0f},
////			{5.0f,6.0f,7.0f,8.0f},
////			{9.0f,10.0f,11.0f,12.0f}
////		};
////		
////		
//		
////		float[][] d = new float[][] {
////			{0.1f,0.2f},
////			{0.3f,0.4f},
////			{0.5f,0.6f}
////		};
//		
//		float[][] dw = new float[W][OW];
//		
//		for(int w = 0;w<W;w++) {
//			for(int ow = 0;ow<OW;ow++) {
//				for(int n = 0;n<N;n++) {
//					dw[w][ow] += x[n][w] * d[n][ow] / N;
//				}
//			}
//		}
//		
//		float[] r = new float[W *OW];
//		
//		float[] x1 = MatrixUtils.transform(x);
//		
//		float[] xt = Transpose.transpose(x1, N, W);
//		
//		float[] d1 = MatrixUtils.transform(d);
//		
//		GPUOP.getInstance().multiplyFloat(W, N, OW, xt, d1, r);
//		
//		r = MatrixOperation.multiplication(r, (1.0f / N));
//		
//		float[][] x1d = MatrixUtils.transform(r,W,OW);
//		
////		float[] x1d = MatrixUtils.transform(x);
////
////		float[] xt2 = Transpose.transpose(x1d, 2, 3);
////		
////		System.out.println(JsonUtils.toJson(xt1));
////		System.out.println(JsonUtils.toJson(xt2));
//		
//		System.out.println(CheckArrayUtils.check(dw, x1d));
		
		TestUtils.dilationTest();
		
	}
	
}
