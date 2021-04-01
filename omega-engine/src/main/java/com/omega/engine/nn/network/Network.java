package com.omega.engine.nn.network;

import java.util.ArrayList;
import java.util.List;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.Layer;

/**
 * base network
 * 
 * @author Administrator
 *
 */
public abstract class Network {
	
	public boolean GRADIENT_CHECK = false;
	
	public int layerCount = 0;
	
	public int trainingTime = 100;

	public int currentTrainingTime = 0;

	public List<Layer> layerList = new ArrayList<Layer>();
	
	public LossFunction lossFunction;
	
	private BaseData trainingData;
	
	public double accuracy = 0.01d;
	
	public double learnRate = 0.01d;
	
	public double errorRate = 0.001d;
	
	public double currentError = 0.0d;
	
	public int inputNum = 0;
	
	public int outputNum = 0;
	
	public abstract void init(BaseData trainData) throws Exception;
	
	public abstract double[] predict(double[] input);
	
	public abstract double[] forward(double[] onceData);
	
	public abstract double[] loss(double[] output,double[] label);
	
	public abstract double[] lossDiff(double[] output,double[] label);
	
	public abstract void back(double[] lossDiff);
	
	public abstract double test(DataSet testData);
	
	public double[] getOuput() {
		// TODO Auto-generated method stub
		return this.layerList.get(this.layerCount - 1).active;
	}
	
	public BaseData getTrainingData() {
		return trainingData;
	}

	public void setTrainingData(BaseData trainingData) {
		this.trainingData = trainingData;
	}
	
	public Layer getLastLayer() {
		return this.layerList.get(this.layerCount - 1);
	}

}
