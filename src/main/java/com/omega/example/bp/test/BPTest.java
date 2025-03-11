package com.omega.example.bp.test;

import com.omega.common.utils.DataLoader;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;

public class BPTest {
	
	public static void bpNetwork_iris() {
		// TODO Auto-generated method stub

		/**
		 * 读取训练数据集
		 */
		String iris_train = "H:/dataset\\iris\\iris.txt";
		
		String iris_test = "H:/dataset\\iris\\iris_test.txt";
		
		String[] labelSet = new String[] {"1","-1"};
		
		DataSet trainData = DataLoader.loalDataByTxt(iris_train, ",", 1, 1, 4, 2, labelSet);
		DataSet testData = DataLoader.loalDataByTxt(iris_test, ",", 1, 1, 4, 2, labelSet);

		System.out.println("train_data:"+JsonUtils.toJson(trainData));
		
		BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
		
		netWork.CUDNN = true;
		
		InputLayer inputLayer = new InputLayer(1,1,4);
		
		FullyLayer hidden1 = new FullyLayer(4, 40);
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(40, 20);
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(20, 2);
		
//		DropoutLayer dropout = new DropoutLayer(0.2f);

		SoftmaxWithCrossEntropyLayer hidden4 = new SoftmaxWithCrossEntropyLayer(2);
		
		netWork.addLayer(inputLayer);
		netWork.addLayer(hidden1);
		netWork.addLayer(active1);
		netWork.addLayer(hidden2);
		netWork.addLayer(active2);
		netWork.addLayer(hidden3);
//		netWork.addLayer(dropout);
		netWork.addLayer(hidden4);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.00001f, 10, LearnRateUpdate.NONE, false);
		
//		    netWork.GRADIENT_CHECK = true;
		
			optimizer.train(trainData);
			
			optimizer.test(testData);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			try {
				CUDAMemoryManager.freeAll();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

	public static void bpNetwork_mnist() {
		// TODO Auto-generated method stub
		/**
		 * 读取训练数据集
		 */
		String mnist_train_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-images.idx3-ubyte";
		
		String mnist_train_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-labels.idx1-ubyte";
		
		String mnist_test_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-images.idx3-ubyte";
		
		String mnist_test_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-labels.idx1-ubyte";
		
		String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
		
		DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, 1, 1 ,784, true);
		
		DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, 1, 1 ,784, true);

		BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
		
		netWork.CUDNN = true;
		
		netWork.learnRate = 0.001f;
		
		int inputCount = (int) (Math.sqrt(794)+10);
		
		InputLayer inputLayer = new InputLayer(1,1,784);
		
		FullyLayer hidden1 = new FullyLayer(784, inputCount, false);
		
		BNLayer bn1 = new BNLayer();
		
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(inputCount, inputCount, false);
		
		BNLayer bn2 = new BNLayer();
		
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(inputCount, inputCount, false);
		
		BNLayer bn3 = new BNLayer();
		
		ReluLayer active3 = new ReluLayer();
		
		FullyLayer hidden4 = new FullyLayer(inputCount, 10);
		
		DropoutLayer dropout = new DropoutLayer(0.2f);
		
		SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

		netWork.addLayer(inputLayer);
		netWork.addLayer(hidden1);
		netWork.addLayer(bn1);
		netWork.addLayer(active1);
		netWork.addLayer(hidden2);
		netWork.addLayer(bn2);
		netWork.addLayer(active2);
		netWork.addLayer(hidden3);
		netWork.addLayer(bn3);
		netWork.addLayer(active3);
		netWork.addLayer(hidden4);
		netWork.addLayer(dropout);
		netWork.addLayer(softmax);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.001f, 128, LearnRateUpdate.NONE, false);

//			netWork.GRADIENT_CHECK = true;
		
			long start = System.nanoTime();
			
			long trainTime = System.nanoTime();
			
			optimizer.train(trainData);
			
			System.out.println("trainTime:"+((System.nanoTime() - trainTime) / 1e9) + "s.");
			
			long testTime = System.nanoTime();
			
			optimizer.test(testData);
			
			System.out.println("testTime:"+((System.nanoTime() - testTime) / 1e9) + "s.");
			
			System.out.println(((System.nanoTime() - start) / 1e9) + "s.");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			try {
				CUDAMemoryManager.freeAll();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public static void main(String[] args) {

		try {

//	    	CUDAModules.initContext();

//	    	bpNetwork_iris();
	    	bpNetwork_mnist();
			
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
