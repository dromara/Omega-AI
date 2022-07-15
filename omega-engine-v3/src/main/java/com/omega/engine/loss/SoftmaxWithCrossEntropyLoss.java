package com.omega.engine.loss;

import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * Cross Entropy loss function
 * 
 * @author Administrator
 *
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 *
 */
public class SoftmaxWithCrossEntropyLoss extends LossFunction {

	public final LossType lossType = LossType.cross_entropy;
	
	private static SoftmaxWithCrossEntropyLoss instance;
	
	private final float eta = 0.000000000001f;
	
	public static SoftmaxWithCrossEntropyLoss operation() {
		if(instance == null) {
			instance = new SoftmaxWithCrossEntropyLoss();
		}
		return instance;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.cross_entropy;
	}

	@Override
	public Blob loss(Blob x, float[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		
		for(int n = 0;n<x.number;n++) {
			for(int o = 0;o<x.width;o++) {
				if(x.maxtir[n][0][0][o] == 0.0d) {
					temp.maxtir[n][0][0][o] = (float) (- label[n][o] * Math.log(eta));
				}else {
					temp.maxtir[n][0][0][o] = (float) (- label[n][o] * Math.log(x.maxtir[n][0][0][o]));
				}
			}
		}

		return temp;
	}

	@Override
	public Blob diff(Blob x, float[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		for(int n = 0;n<x.number;n++) {
			for(int o = 0;o<x.width;o++) {
				temp.maxtir[n][0][0][o] = - label[n][o] / x.maxtir[n][0][0][o];
			}
		}
		return temp;
	}
	
	public static void main(String[] args) {
		float[][] x = new float[][] {{0.2f,0.3f,0.5f},{0.1f,0.1f,0.8f},{0.3f,0.1f,0.6f},{0.9f,0.01f,0.09f}};
		Blob xb = Blobs.blob(4, 1, 1, 3, x);
		float[][] label = new float[][] {{0,1,0},{1,0,0},{1,0,0},{0,0,1}};
		float error = SoftmaxWithCrossEntropyLoss.operation().gradientCheck(xb,label);
		System.out.println("error:"+error);
	}

}
