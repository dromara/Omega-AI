package com.omega.engine.nn.network;

import java.util.ArrayList;
import java.util.List;

import com.omega.engine.loss.LossFunction;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.model.NetworkInit;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;

/**
 * base network
 * 
 * @author Administrator
 *
 */
public abstract class Network {
	
	private int threadNum = 8;
	
	public NetworkType networkType;
	
	public UpdaterType updater = UpdaterType.none;
	
	public RunModel RUN_MODEL = RunModel.TRAIN;
	
	public boolean GRADIENT_CHECK = false;
	
	public int layerCount = 0;
	
	public int trainingTime = 100;

	public int currentTrainingTime = 0;

	public List<Layer> layerList = new ArrayList<Layer>();
	
	public LossFunction lossFunction;
	
	public Blob inputData;
	
	public Blob lossDiff;
	
	public double accuracy = 0.01d;
	
	public double learnRate = 0.01d;
	
	public double errorRate = 0.001d;
	
	public double currentError = 0.0d;
	
	/**
	 * batch number
	 */
	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;
	
	public int width = 0;
	
	public int oChannel = 0;
	
	public int oHeight = 0;
	
	public int oWidth = 0;
	
	public abstract void init() throws Exception;
	
	public abstract Blob predict(Blob input);
	
	public abstract Blob forward(Blob input);
	
	public abstract void back(Blob lossDiff);
	
	public abstract Blob loss(Blob output,double[][] label);
	
	public abstract Blob lossDiff(Blob output,double[][] label);
	
	public abstract NetworkType getNetworkType();
	
	public void setNumber(int number) {
		this.number = number;
	}
	
	public void setInputData(Blob inputData) {
		this.number = inputData.number;
		this.inputData = inputData;
	}

	public void setLossDiff(Blob lossDiff) {
		this.lossDiff = lossDiff;
		this.getLastLayer().setDelta(this.lossDiff);
	}

	public Blob getOuput() {
		// TODO Auto-generated method stub
		return this.getLastLayer().getOutput();
	}
	
	public Layer getLastLayer() {
		return this.layerList.get(this.layerCount - 1);
	}
	
	public Layer getPreLayer(int index) {

		if(index > 0 && index < this.layerCount) {
			return this.layerList.get(index - 1);
		}
		
		return null;
	}
	
	public Layer getNextLayer(int index) {
		
		if(index > 0 && index < this.layerCount - 1) {
			return this.layerList.get(index + 1);
		}
		
		return null;
	}
	
	public void addLayer(Layer layer) {
		layer.setNetwork(this);
		layer.setIndex(this.layerList.size());
		layer.setUpdater(UpdaterFactory.create(this.updater));
		this.layerList.add(layer);
	}
	
	public NetworkInit save() {
		NetworkInit init = new NetworkInit(this);
		for (Layer layer:this.layerList) {
			init.getLayers().add(layer.save());
		}
		return init;
	}

	public int getThreadNum() {
		return threadNum;
	}

	public void setThreadNum(int threadNum) {
		this.threadNum = threadNum;
	}
	
}
