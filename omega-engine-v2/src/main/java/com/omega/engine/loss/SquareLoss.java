package com.omega.engine.loss;

import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class SquareLoss extends LossFunction {
	
	public final LossType lossType = LossType.square_loss;
	
	private static SquareLoss instance;
	
	public static SquareLoss operation() {
		if(instance == null) {
			instance = new SquareLoss();
		}
		return instance;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.square_loss;
	}
	
	@Override
	public Blob loss(Blob x, double[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		for(int n = 0;n<x.number;n++) {
			for(int w = 0;w<x.width;w++) {
				temp.maxtir[n][0][0][w] = (x.maxtir[n][0][0][w] - label[n][w]) * (x.maxtir[n][0][0][w] - label[n][w]) / 2;;
			}
		}
		return temp;
	}

	@Override
	public Blob diff(Blob x, double[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		for(int n = 0;n<x.number;n++) {
			for(int w = 0;w<x.width;w++) {
				temp.maxtir[n][0][0][w] = x.maxtir[n][0][0][w] - label[n][w];
			}
		}
		return temp;
	}
	
	public static void main(String[] args) {
		double[][] x = new double[][] {{0.1,-0.03,1.23,-0.4,0.1,-0.12,0.001,0.002}};
		Blob xb = Blobs.blob(1, 1, 1, 8, x);
		double[][] label = new double[][] {{0.1,-0.01,0.022,-0.4,0.803,-0.12,0.001,0.001}};
		double error = SquareLoss.operation().gradientCheck(xb,label);
		System.out.println("error:"+error);
	}

}
