package com.omega.engine.ops;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.layer.normalization.BNType;

public class BNBackward extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 159880198146081977L;

	private int start = 0;
	
	private int end = 0;
	
	private BNLayer layer;
	
	private float[] dvar;
	
	private float[] dmu;
	
	
	public BNBackward(BNLayer layer,float[] dvar,float[] dmu,int start,int end) {
		this.layer = layer;
		this.start = start;
		this.end = end;
		this.dvar = dvar;
		this.dmu = dmu;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= layer.number / 8) {
			
			exc();

		} else {

			int mid = (start + end + 1) >>> 1;
			BNBackward left = new BNBackward(layer, dvar, dmu, start, mid - 1);
			BNBackward right = new BNBackward(layer, dvar, dmu, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}

	public void exc() {

		for(int n = start;n<=end;n++) {
			for(int c = 0;c<layer.oChannel;c++) {
				for(int h = 0;h<layer.oHeight;h++) {
					for(int w = 0;w<layer.oWidth;w++) {
						int ci = w;
						int bn = layer.number;
						if(layer.bnType == BNType.conv_bn) {
							ci = c;
							bn = layer.number * layer.oHeight * layer.oWidth;
						}
						// deltaGama = ∑ deta * z
						layer.deltaGama[ci] += layer.delta.maxtir[n][c][h][w] * layer.z.maxtir[n][c][h][w];
						// deltaBeta = ∑ deta
						layer.deltaBeta[ci] += layer.delta.maxtir[n][c][h][w];
						// dxhat = deta * gama
						layer.diff.maxtir[n][c][h][w] = layer.delta.maxtir[n][c][h][w] * layer.gama[ci];
						// dstd = ∑ dxhat * (xi - mean) * -1/2 * (std + eta)^-3/2
						dvar[ci] += layer.diff.maxtir[n][c][h][w] * (layer.input.maxtir[n][c][h][w] - layer.mean[ci]) * -0.5f * Math.pow(layer.std[ci], -1.5);
						// dmean = ∑ dxhat * -1 / (std + eta)^1/2 + dstd * (∑ -2 * (x - mean)) / n
						dmu[ci] +=  -1.0f * layer.diff.maxtir[n][c][h][w] /layer. std[ci] + -2.0f * (layer.input.maxtir[n][c][h][w] - layer.mean[ci]) / bn;
		
					}
				}
			}
		}

	}
	
}
