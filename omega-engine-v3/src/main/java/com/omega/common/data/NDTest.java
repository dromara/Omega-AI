package com.omega.common.data;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import com.omega.common.utils.Im2col;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;

public class NDTest {
	
	public static void main(String args[]) {
		
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		int pad = 0;
		
		int N = 4;
		int C = 3;
		int H = 3;
		int W = 3;
		
		float[][][][] x = RandomUtils.gaussianRandom(N, C, H, W, 0.1f);
		
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

		System.out.println("==========================");

		INDArray img = Nd4j.create(x);
		
		INDArray colN =  Convolution.im2col(img, fh, fw, stride, stride, 0, 0, 1, 1, false);
		
		System.out.println(JsonUtils.toJson(colN.shape()));
		
		INDArray im2col2d = Shape.newShapeNoCopy(colN, new long[]{
				N * oh * ow, fh * fw * C
        }, false );
		
		System.out.println(JsonUtils.toJson(im2col2d.shape()));
		
		PrintUtils.printImage(im2col2d.toFloatMatrix());
		
	}
	
}
