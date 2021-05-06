package com.omega.engine.updater;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;

/**
 * Momentum
 * 
 * Vdw = beta * Vdw + (1 - beta) * dw
 * Vdb = beta * Vdb + (1 - beta) * db
 * W = W - learnRate * Vdw
 * b = b - learnRate * Vdb
 * 
 * @author Administrator
 *
 */
public class Momentum extends Updater {
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(this.vdw == null || this.vdb == null) {
			this.vdw = MatrixOperation.zero(layer.inputNum, layer.outputNum);
			this.vdb = MatrixOperation.zero(layer.outputNum);
		}
		
		/**
		 * Vdw = beta * Vdw + (1 - beta) * dw
		 */
		for(int i = 0;i<this.vdw.length;i++) {
			for(int j = 0;j<this.vdw[i].length;j++) {
				this.vdw[i][j] = this.beta * this.vdw[i][j] + (1 - this.beta) * layer.deltaW[i][j];
			}
		}
		
		/**
		 * Vdb = beta * Vdb + (1 - beta) * db
		 */
		for(int i = 0;i<this.vdb.length;i++) {
			this.vdb[i] = this.beta * this.vdb[i] + (1 - this.beta) * layer.delta[i];
		}
		
		/**
		 * W = W - learnRate * Vdw
		 */
		layer.weight = MatrixOperation.subtraction(layer.weight, MatrixOperation.multiplication(this.vdw, layer.learnRate));
		
		/**
		 * b = b - learnRate * Vdb
		 */
		layer.bias = MatrixOperation.subtraction(layer.bias, MatrixOperation.multiplication(this.vdb, layer.learnRate));
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
		if(this.vdmw == null || this.vdmb == null) {
			this.vdmw = MatrixOperation.zero(conv.channel, conv.kernelNum, conv.kHeight, conv.kWidth);
			this.vdmb = MatrixOperation.zero(conv.kernelNum);
		}
		
		/**
		 * Vdw = beta * Vdw + (1 - beta) * dw
		 */
		for(int c = 0;c<this.vdmw.length;c++) {
			for(int k = 0;k<this.vdmw[c].length;k++) {
				for(int h = 0;h<this.vdmw[c][k].length;h++) {
					for(int w = 0;w<this.vdmw[c][k][h].length;w++) {
						this.vdmw[c][k][h][w] = this.beta * this.vdmw[c][k][h][w] + (1 - this.beta) * conv.deltaW[c][k][h][w];
					}
				}
			}
		}
		
		/**
		 * Vdb = beta * Vdb + (1 - beta) * db
		 */
		for(int k = 0;k<this.vdmb.length;k++) {
			this.vdmb[k] = this.beta * this.vdmb[k] + (1 - this.beta) * conv.deltaB[k];
		}
		
		/**
		 * W = W - learnRate * Vdw
		 */
		conv.kernel = MatrixOperation.subtraction(conv.kernel, MatrixOperation.multiplication(this.vdmw, conv.learnRate));
		
		/**
		 * b = b - learnRate * Vdb
		 */
		conv.bias = MatrixOperation.subtraction(conv.bias, MatrixOperation.multiplication(this.vdmb, conv.learnRate));
	}

}
