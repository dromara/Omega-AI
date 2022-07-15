package com.omega.engine.updater;

import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNLayer;

/**
 * Adam Updater
 * @author Administrator
 *
 */
public class Adam extends Updater {
	
	private float beta1 = 0.9f;
	
	private float beta2 = 0.999f;
	
	private float eta = 10e-8f;
	
	private float[][] mw;
	
	private float[][] vw;
	
	private float[] mb;
	
	private float[] vb;
	
	private float[][][][] mmw;
	
	private float[][][][] vmw;

	private float[] mgama;
	
	private float[] vgama;
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(this.mw == null || this.vw == null) {
			this.mw = MatrixUtils.zero(layer.width, layer.oWidth);
			this.vw = MatrixUtils.zero(layer.width, layer.oWidth);
			if(layer.hasBias) {
				this.mb = MatrixUtils.zero(layer.oWidth);
				this.vb = MatrixUtils.zero(layer.oWidth);
			}
		}
		
		this.updateW(layer);
		
		if(layer.hasBias) {
			
			this.updateB(layer);
			
		}
		
	}

	@Override
	public void updateForMatrix(Layer layer) {
		// TODO Auto-generated method stub

		if(!layer.getLayerType().equals(LayerType.conv)) {
			throw new RuntimeException("this function param must be conv layer.");
		}
		
		ConvolutionLayer conv = (ConvolutionLayer) layer;
		
		/**
		 * init
		 */
		if(this.mmw == null || this.vmw == null) {
			
			this.mmw = MatrixUtils.zero(conv.kernelNum, conv.channel, conv.kHeight, conv.kWidth);
			this.vmw = MatrixUtils.zero(conv.kernelNum, conv.channel, conv.kHeight, conv.kWidth);

			if(layer.hasBias) {
				this.mb = MatrixUtils.zero(conv.kernelNum);
				this.vb = MatrixUtils.zero(conv.kernelNum);
			}
			
		}

		this.updateW(conv);
		
		if(conv.hasBias) {

			this.updateB(conv);
			
		}

	}

	@Override
	public void updateForBN(BNLayer layer) {
		// TODO Auto-generated method stub
		
//		this.lr = layer.learnRate / layer.number;
//		System.out.println(layer.learnRate);
		/**
		 * init
		 */
		if(this.mgama == null || this.vgama == null) {
			this.mgama = MatrixUtils.zero(layer.deltaGama.length);
			this.vgama = MatrixUtils.zero(layer.deltaGama.length);
			this.mb = MatrixUtils.zero(layer.deltaBeta.length);
			this.vb = MatrixUtils.zero(layer.deltaBeta.length);
		}
		
		for(int i = 0;i<layer.deltaGama.length;i++){
			this.mgama[i] = this.beta1 * this.mgama[i] + (1 - this.beta1) * layer.deltaGama[i];
			this.vgama[i] = this.beta2 * this.vgama[i] + (1 - this.beta2) * layer.deltaGama[i] * layer.deltaGama[i];
			float mhat = this.mgama[i] / (1 - beta1);
			float vhat = this.vgama[i] / (1 - beta2);
			layer.gama[i] = layer.gama[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
		for(int i = 0;i<layer.deltaBeta.length;i++){
			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * layer.deltaBeta[i];
			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * layer.deltaBeta[i] * layer.deltaBeta[i];
			float mhat = this.mb[i] / (1 - beta1);
			float vhat = this.vb[i] / (1 - beta2);
			layer.beta[i] = layer.beta[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.adam;
	}
	
	/**
	 * mt = beta1 * mt-1 + (1 - beta1) * gt
	 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
	 * mhat = mt / (1 - beta1)
	 * vhat = vt / (1 - beta2)
	 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
	 */
	public void updateW(ConvolutionLayer conv) {
		
		float[][][][] deltaW = conv.deltaW;
		
		int N = deltaW.length;
		int C = deltaW[0].length;
		int H = deltaW[0][0].length;
		int W = deltaW[0][0][0].length;
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<N;n++){
			
			final int index = n;
			
			workers.add(new Task<Object>(index) {
				
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								mmw[index][c][h][w] = beta1 * mmw[index][c][h][w] + (1 - beta1) * deltaW[index][c][h][w];
								vmw[index][c][h][w] = beta2 * vmw[index][c][h][w] + (1 - beta2) * deltaW[index][c][h][w] * deltaW[index][c][h][w];
								float mhat = mmw[index][c][h][w] / (1 - beta1);
								float vhat = vmw[index][c][h][w] / (1 - beta2);
								
								conv.kernel[index][c][h][w] = conv.kernel[index][c][h][w] - conv.learnRate * mhat / ((float)Math.sqrt(vhat) + eta);
							}
						}
					}
					return null;
				}
			});
			
		}
		
		TaskEngine.getInstance(8).dispatchTask(workers);
		
	}
	
	/**
	 * mt = beta1 * mt-1 + (1 - beta1) * gt
	 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
	 * mhat = mt / (1 - beta1)
	 * vhat = vt / (1 - beta2)
	 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
	 */
	public void updateW(Layer layer) {
		
		float[][] deltaW = layer.deltaW;
		
		int H = deltaW.length;
		int W = deltaW[0].length;

		for(int h = 0;h<H;h++) {
			for(int w = 0;w<W;w++) {
				this.mw[h][w] = this.beta1 * this.mw[h][w] + (1 - this.beta1) * deltaW[h][w];
				this.vw[h][w] = this.beta2 * this.vw[h][w] + (1 - this.beta2) * deltaW[h][w] * deltaW[h][w];
				float mhat = this.mw[h][w] / (1 - beta1);
				float vhat = this.vw[h][w] / (1 - beta2);
				layer.weight[h][w] = layer.weight[h][w] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
			}
		}

	}
	
	/**
	 * mt = beta1 * mt-1 + (1 - beta1) * gt
	 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
	 * mhat = mt / (1 - beta1)
	 * vhat = vt / (1 - beta2)
	 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
	 */
	public void updateB(Layer layer) {
		
		float[] deltaB = layer.deltaB;

		for(int i = 0;i<deltaB.length;i++){
			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * deltaB[i];
			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * deltaB[i] * deltaB[i];
			float mhat = this.mb[i] / (1 - beta1);
			float vhat = this.vb[i] / (1 - beta2);
			layer.bias[i] = layer.bias[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
	}
	
}
