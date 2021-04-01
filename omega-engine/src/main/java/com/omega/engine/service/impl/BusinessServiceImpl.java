package com.omega.engine.service.impl;

import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;

import com.omega.common.utils.DataLoader;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.active.Relu;
import com.omega.engine.active.Sigmoid;
import com.omega.engine.loss.CrossEntropyLoss;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.SoftmaxLayer;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.service.BusinessService;
import com.omega.engine.updater.Momentum;

@Service
public class BusinessServiceImpl implements BusinessService {

//	@Autowired
//	private TaskEngine trainEngine;
	
	@Override
	public void bpNetwork_iris() {
		// TODO Auto-generated method stub
		

//		double[][] dataInput = {
//				{5.5,2.6,4.4,1.2},{6.1,3.0,4.6,1.4},{5.8,2.6,4.0,1.2},{5.0,2.3,3.3,1.0},{5.6,2.7,4.2,1.3},{5.7,3.0,4.2,1.2},{5.7,2.9,4.2,1.3},{6.2,2.9,4.3,1.3},{5.1,2.5,3.0,1.1},{5.7,2.8,4.1,1.3},
//				{5.0,3.5,1.3,0.3},{4.5,2.3,1.3,0.3},{4.4,3.2,1.3,0.2},{5.0,3.5,1.6,0.6},{5.1,3.8,1.9,0.4},{4.8,3.0,1.4,0.3},{5.1,3.8,1.6,0.2},{4.6,3.2,1.4,0.2},{5.3,3.7,1.5,0.2},{5.0,3.3,1.4,0.2}
//			   };
//		
//		double[][] dataLabel = {
//								{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},
//								{1},{1},{1},{1},{1},{1},{1},{1},{1},{1}
//							   };
//		
//		double[][] dataInput = {
//				{0,0},{0,1},{1,1},{1,0}
//			   };
//		
//		double[][] dataLabel = {
//								{0},{1},{0},{1}
//							   };
//		
//		DataSet trainData = new DataSet(20, 4, 1, dataInput, dataLabel);
		
		/**
		 * 读取训练数据集
		 */
		String iris_train = "E:\\dataset\\iris\\iris.txt";
		
		String iris_test = "E:\\dataset\\iris\\iris_test.txt";
		
		String[] labelSet = new String[] {"1","-1"};
		
		DataSet trainData = DataLoader.loalDataByTxt(iris_train, ",", 4, 2, labelSet);
		DataSet testData = DataLoader.loalDataByTxt(iris_test, ",", 4, 2, labelSet);
		
//		DataSet trainData = DataLoader.loalDataByTxt(iris_train, ",", 4, 2);
//		DataSet testData = DataLoader.loalDataByTxt(iris_test, ",", 4, 2);
		
		System.out.println("train_data:"+JsonUtils.toJson(trainData));
	
		List<Layer> layers = new ArrayList<Layer>();
		
		InputLayer inputLayer = new InputLayer(4);
		
		FullyLayer hidden1 = new FullyLayer(4, 40, new Relu(),new Momentum());
		
		FullyLayer hidden2 = new FullyLayer(40, 20, new Relu(),new Momentum());
		
		FullyLayer hidden3 = new FullyLayer(20, 2, new Sigmoid(),new Momentum());
		
		SoftmaxLayer hidden4 = new SoftmaxLayer(2, 2,new Momentum());
		
		layers.add(inputLayer);
		layers.add(hidden1);
		layers.add(hidden2);
		layers.add(hidden3);
		layers.add(hidden4);
		
		
		BPNetwork netWork = new BPNetwork(layers,new CrossEntropyLoss());
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 2000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);
		
		MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 100000, 0.0001d, 10, LearnRateUpdate.NONE);
		
//		netWork.GRADIENT_CHECK = true;
		
		try {
			
			netWork.init(trainData);
			
			optimizer.train();
			
			netWork.test(testData);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	public void bpNetwork_mnist() {
		// TODO Auto-generated method stub
		/**
		 * 读取训练数据集
		 */
		String mnist_train_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-images.idx3-ubyte";
		
		String mnist_train_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-labels.idx1-ubyte";
		
		String mnist_test_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-images.idx3-ubyte";
		
		String mnist_test_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-labels.idx1-ubyte";
		
		String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
		
		DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, true);
		
		DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, true);

		System.out.println("train_data[dataSize:"+trainData.dataSize+",inputSize:"+trainData.inputSize+"]");
		
		int inputCount = (int) (Math.sqrt(794)+10);

		List<Layer> layers = new ArrayList<Layer>();
		
		InputLayer inputLayer = new InputLayer(784);
		
		FullyLayer hidden1 = new FullyLayer(784, inputCount, new Relu(),new Momentum());
		
//		hidden1.learnRate = 0.001d;
		
		FullyLayer hidden2 = new FullyLayer(inputCount, inputCount, new Relu(),new Momentum());
		
//		hidden2.learnRate = 0.001d;
		
		FullyLayer hidden3 = new FullyLayer(inputCount, inputCount, new Relu(),new Momentum());
		
//		hidden3.learnRate = 0.001d;
		
		FullyLayer hidden4 = new FullyLayer(inputCount, 10, new Relu(),new Momentum());
		
//		hidden4.learnRate = 0.001d;
		
		SoftmaxWithCrossEntropyLayer hidden5 = new SoftmaxWithCrossEntropyLayer(10,new Momentum());
		
//		hidden5.learnRate = 0.001d;
		
		layers.add(inputLayer);
		layers.add(hidden1);
		layers.add(hidden2);
		layers.add(hidden3);
		layers.add(hidden4);
		layers.add(hidden5);
		
		BPNetwork netWork = new BPNetwork(layers,new CrossEntropyLoss());
		
		netWork.learnRate = 0.01d;
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 20000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);
		
		MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 50000, 0.0000001d, 64, LearnRateUpdate.NONE);
		
		//启动多线程
//		optimizer.setTrainEngine(trainEngine);
		
//		netWork.GRADIENT_CHECK = true;
		
		try {
			
			long start = System.currentTimeMillis();
			
			netWork.init(trainData);
			
			optimizer.train();
			
			netWork.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public void cnnNetwork_mnist() {
		// TODO Auto-generated method stub
		/**
		 * 读取训练数据集
		 */
		String mnist_train_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-images.idx3-ubyte";
		
		String mnist_train_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-labels.idx1-ubyte";
		
		String mnist_test_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-images.idx3-ubyte";
		
		String mnist_test_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-labels.idx1-ubyte";
		
		String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
		
		DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet,true);
		
		DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet,true);

		System.out.println("train_data[dataSize:"+trainData.dataSize+",inputSize:"+trainData.inputSize+"]");
		
		int channel = 1;
		
		int height = 28;
		
		int width = 28;
		
		List<Layer> layers = new ArrayList<Layer>();
		
		InputLayer inputLayer = new InputLayer(784, channel, height, width);
		
		ConvolutionLayer conv1 = new ConvolutionLayer(channel, 32, width, height, 5, 5, 4, 1, new Relu(), null);
		
		PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING, null);
		
		ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 20, pool1.oWidth, pool1.oHeight, 3, 3, 0, 1, new Relu(), null);
		
		PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING, null);

		int inputCount = (int) (Math.sqrt((pool2.oChannel * pool2.oWidth * pool2.oHeight) + 10)+10);
		
		FullyLayer full1 = new FullyLayer(pool2.oChannel * pool2.oWidth * pool2.oHeight, inputCount, new Relu(),new Momentum());

		FullyLayer full2 = new FullyLayer(inputCount, 10, new Relu(),new Momentum());

		SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10,new Momentum());

		layers.add(inputLayer);
		layers.add(conv1);
		layers.add(pool1);
		layers.add(conv2);
		layers.add(pool2);
		layers.add(full1);
		layers.add(full2);
		layers.add(softmax);
		
		CNN netWork = new CNN(layers, new CrossEntropyLoss());
		
		netWork.learnRate = 0.01d;
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 20000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);
		
		MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1, 0.0000001d, 64, LearnRateUpdate.NONE);
		
		//启动多线程
//		optimizer.setTrainEngine(trainEngine);
		
//		netWork.GRADIENT_CHECK = true;
		
		try {
			
			long start = System.currentTimeMillis();
			
			netWork.init(trainData);
			
			optimizer.train();
			
			netWork.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		BusinessServiceImpl bs = new BusinessServiceImpl();
//		bs.bpNetwork_iris();
//		bs.bpNetwork_mnist();
		bs.cnnNetwork_mnist();
	}

}
