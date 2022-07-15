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
	public Blob loss(Blob x, float[][] label) {
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
	public Blob diff(Blob x, float[][] label) {
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
		float[][] x = new float[][] {{0.1f,-0.03f,1.23f,-0.4f,0.1f,-0.12f,0.001f,0.002f}};
		Blob xb = Blobs.blob(1, 1, 1, 8, x);
		float[][] label = new float[][] {{0.1f,-0.01f,0.022f,-0.4f,0.803f,-0.12f,0.001f,0.001f}};
		float error = SquareLoss.operation().gradientCheck(xb,label);
		System.out.println("error:"+error);
	}

}
