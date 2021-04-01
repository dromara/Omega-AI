package com.omega.engine.nn.layer;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.active.ActiveFunction;
import com.omega.engine.active.ActiveType;
import com.omega.engine.updater.Updater;

public abstract class Layer {
	
	public boolean GRADIENT_CHECK = false;
	
	public int index = 0;
	
	public int inputNum;
	
	public int outputNum;
	
	public double[] input;
	
	public double[] output;
	
	public double[] active;
	
	public double[] diff;
	
	public double[] delta;
	
	public double[][] deltaW;
	
	public double[] nextDiff;
	
	public double[][] weight;
	
	public double[] bias;
	
	public double lambda = 0.01d;
	
	public double learnRate = 0.001d;
	
	public double eta = 0.00001d;
	
	public LayerType layerType;
	
	public ActiveFunction activeFunction;
	
	public Updater updater;
	
	public int[] inputShape = new int[] { 0, 0, 0, 0};
	
	public int[] outputShape = new int[] { 0, 0, 0, 0};

	public abstract void init();
	
	public abstract void initParam();

	public abstract void output();
	
	public abstract double[] getOutput();
	
	public abstract void active();
	
	public abstract double[] activeTemp(double[] output);
	
	public abstract void diff();
	
	public abstract double[] diffTemp();
	
	public abstract void forward();
	
	public abstract void back();
	
	public abstract void update();
	
	public abstract LayerType getLayerType();
	
	public void setActive(ActiveFunction activeFunction) {
		this.activeFunction = activeFunction;
	}

	public void setUpdater(Updater updater) {
		this.updater = updater;
	}
	
	/**
	 * 
	 * @Title: gradientCheck
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 * gradientCheck:
	 * (f(x + eta) - f(x - eta)) / (2 * eta) ≈ f'(x)
	 */
	public double gradientCheck() {
		System.out.println("*******************" + this.index + "="+this.getLayerType()+" layer********************");
		this.output();
		double[] output1 = MatrixOperation.add(this.getOutput(),eta);
		double[] output2 = MatrixOperation.subtraction(this.getOutput(),eta);
		double[] f1 = this.activeTemp(output1);		
		double[] f2 = this.activeTemp(output2);
		
		double[] gradientCheck = MatrixOperation.subtraction(f1, f2);
		
		gradientCheck = MatrixOperation.division(gradientCheck, 2 * this.eta);
		
		if(this.activeFunction!=null && this.activeFunction.activeType == ActiveType.relu) {
			/**
			 * fix zero output diff is zero.
			 */
			for(int i = 0;i<gradientCheck.length;i++) {
				if(this.getOutput()[i] == 0) {
					gradientCheck[i] = 0;
				}
			}
		}
		
		double[] currentDiff = this.diffTemp();

		System.out.println("currentDiff:"+JsonUtils.toJson(currentDiff));

		System.out.println("gradientCheck:"+JsonUtils.toJson(gradientCheck));
		
		double finalGCError = 0.0d;
		
		if(this.getLayerType()!=LayerType.pooling) {
			double[] error = MatrixOperation.subtractionP(currentDiff, gradientCheck);
			System.out.println(JsonUtils.toJson(error));
			finalGCError = MatrixOperation.sum(error);
		}
		
		System.out.println("finalGCError:"+finalGCError);
		return 0.0d;
	}
	
}
