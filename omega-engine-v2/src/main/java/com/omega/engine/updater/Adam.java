package com.omega.engine.updater;

import com.omega.common.utils.MatrixOperation;
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
	
	private double beta1 = 0.9d;
	
	private double beta2 = 0.999d;
	
	private double eta = 10e-8;
	
	private double[][] mw;
	
	private double[][] vw;
	
	private double[] mb;
	
	private double[] vb;
	
	private double[][][][] mmw;
	
	private double[][][][] vmw;

	private double[] mgama;
	
	private double[] vgama;
	
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
		
		/**
		 * mt = beta1 * mt-1 + (1 - beta1) * gt
		 */
		this.mw = MatrixOperation.add(MatrixOperation.multiplication(this.mw, this.beta1),MatrixOperation.multiplication(layer.deltaW,1 - beta1));
		
		/**
		 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
		 */
		this.vw = MatrixOperation.add(MatrixOperation.multiplication(this.vw, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(layer.deltaW, 2),1 - beta2));
		
		/**
		 * mhat = mt / (1 - beta1)
		 */
		double[][] mhatw = MatrixOperation.division(this.mw, (1 - beta1));
		
		/**
		 * vhat = vt / (1 - beta2)
		 */
		double[][] vhatw = MatrixOperation.division(this.vw, (1 - beta2));

		/**
		 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
		 */
		layer.weight = MatrixOperation.subtraction(layer.weight, MatrixOperation.multiplication(MatrixOperation.division(mhatw, (MatrixOperation.add(MatrixOperation.sqrt(vhatw), this.eta))), layer.learnRate));
		
		if(layer.hasBias) {
			
			/**
			 * mt = beta1 * mt-1 + (1 - beta1) * gt
			 */
			this.mb = MatrixOperation.add(MatrixOperation.multiplication(this.mb, this.beta1),MatrixOperation.multiplication(layer.deltaB,1 - beta1));
			
			/**
			 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
			 */
			this.vb = MatrixOperation.add(MatrixOperation.multiplication(this.vb, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(layer.deltaB, 2),1 - beta2));
			
			/**
			 * mhat = mt / (1 - beta1)
			 */
			double[] mhatb = MatrixOperation.division(this.mb, (1 - beta1));
			
			/**
			 * vhat = vt / (1 - beta2)
			 */
			double[] vhatb = MatrixOperation.division(this.vb, (1 - beta2));

			/**
			 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
			 */
			layer.bias = MatrixOperation.subtraction(layer.bias, MatrixOperation.multiplication(MatrixOperation.division(mhatb, (MatrixOperation.add(MatrixOperation.sqrt(vhatb), this.eta))), layer.learnRate));
	
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
			
			this.mmw = MatrixUtils.zero(conv.channel, conv.kernelNum, conv.kHeight, conv.kWidth);
			this.vmw = MatrixUtils.zero(conv.channel, conv.kernelNum, conv.kHeight, conv.kWidth);

			if(layer.hasBias) {
				this.mb = MatrixUtils.zero(conv.kernelNum);
				this.vb = MatrixUtils.zero(conv.kernelNum);
			}
			
		}
			
		/**
		 * mt = beta1 * mt-1 + (1 - beta1) * gt
		 */
		this.mmw = MatrixOperation.add(MatrixOperation.multiplication(this.mmw, this.beta1),MatrixOperation.multiplication(conv.deltaW,1 - beta1));
		
		/**
		 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
		 */
		this.vmw = MatrixOperation.add(MatrixOperation.multiplication(this.vmw, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(conv.deltaW, 2),1 - beta2));
		
		/**
		 * mhat = mt / (1 - beta1)
		 */
		double[][][][] mhatw = MatrixOperation.division(this.mmw, (1 - beta1));
		
		/**
		 * vhat = vt / (1 - beta2)
		 */
		double[][][][] vhatw = MatrixOperation.division(this.vmw, (1 - beta2));

		/**
		 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
		 */
		conv.kernel = MatrixOperation.subtraction(conv.kernel, MatrixOperation.multiplication(MatrixOperation.division(mhatw, (MatrixOperation.add(MatrixOperation.sqrt(vhatw), this.eta))), conv.learnRate));

		if(conv.hasBias) {
			/**
			 * mt = beta1 * mt-1 + (1 - beta1) * gt
			 */
			this.mb = MatrixOperation.add(MatrixOperation.multiplication(this.mb, this.beta1),MatrixOperation.multiplication(conv.deltaB,1 - beta1));
			
			/**
			 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
			 */
			this.vb = MatrixOperation.add(MatrixOperation.multiplication(this.vb, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(conv.deltaB, 2),1 - beta2));
			
			/**
			 * mhat = mt / (1 - beta1)
			 */
			double[] mhatb = MatrixOperation.division(this.mb, (1 - beta1));
			
			/**
			 * vhat = vt / (1 - beta2)
			 */
			double[] vhatb = MatrixOperation.division(this.vb, (1 - beta2));

			/**
			 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
			 */
			conv.bias = MatrixOperation.subtraction(conv.bias, MatrixOperation.multiplication(MatrixOperation.division(mhatb, (MatrixOperation.add(MatrixOperation.sqrt(vhatb), this.eta))), conv.learnRate));

		}

	}

	@Override
	public void updateForBN(BNLayer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(this.mgama == null || this.vgama == null) {
			this.mgama = MatrixUtils.zero(layer.deltaGama.length);
			this.vgama = MatrixUtils.zero(layer.deltaGama.length);
			this.mb = MatrixUtils.zero(layer.deltaBeta.length);
			this.vb = MatrixUtils.zero(layer.deltaBeta.length);
		}
		
		/**
		 * mt = beta1 * mt-1 + (1 - beta1) * gt
		 */
		this.mgama = MatrixOperation.add(MatrixOperation.multiplication(this.mgama, this.beta1),MatrixOperation.multiplication(layer.deltaGama,1 - beta1));
		
		/**
		 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
		 */
		this.vgama = MatrixOperation.add(MatrixOperation.multiplication(this.vgama, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(layer.deltaGama, 2),1 - beta2));
		
		/**
		 * mhat = mt / (1 - beta1)
		 */
		double[] mhatw = MatrixOperation.division(this.mgama, (1 - beta1));
		
		/**
		 * vhat = vt / (1 - beta2)
		 */
		double[] vhatw = MatrixOperation.division(this.vgama, (1 - beta2));

		/**
		 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
		 */
		layer.gama = MatrixOperation.subtraction(layer.gama, MatrixOperation.multiplication(MatrixOperation.division(mhatw, (MatrixOperation.add(MatrixOperation.sqrt(vhatw), this.eta))), layer.learnRate));
		

		/**
		 * mt = beta1 * mt-1 + (1 - beta1) * gt
		 */
		this.mb = MatrixOperation.add(MatrixOperation.multiplication(this.mb, this.beta1),MatrixOperation.multiplication(layer.deltaBeta,1 - beta1));
		
		/**
		 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
		 */
		this.vb = MatrixOperation.add(MatrixOperation.multiplication(this.vb, this.beta2),MatrixOperation.multiplication(MatrixOperation.pow(layer.deltaBeta, 2),1 - beta2));
		
		/**
		 * mhat = mt / (1 - beta1)
		 */
		double[] mhatb = MatrixOperation.division(this.mb, (1 - beta1));
		
		/**
		 * vhat = vt / (1 - beta2)
		 */
		double[] vhatb = MatrixOperation.division(this.vb, (1 - beta2));

		/**
		 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
		 */
		layer.beta = MatrixOperation.subtraction(layer.beta, MatrixOperation.multiplication(MatrixOperation.division(mhatb, (MatrixOperation.add(MatrixOperation.sqrt(vhatb), this.eta))), layer.learnRate));

	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.adam;
	}

}
