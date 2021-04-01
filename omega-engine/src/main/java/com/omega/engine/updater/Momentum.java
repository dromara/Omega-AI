package com.omega.engine.updater;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.Layer;

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

}
