package com.omega.engine.service.impl;

import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import com.omega.common.utils.DataLoader;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.controller.TrainTask;
import com.omega.engine.loss.CrossEntropyLoss;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.BasicBlockLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.model.NetworkInit;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.service.BusinessService;
import com.omega.engine.updater.UpdaterType;

@Service
public class BusinessServiceImpl implements BusinessService {
	
	/*
	 * @Autowired private NetworksDataBase dataBase;
	 */

	@Override
	public void bpNetwork_iris() {
		// TODO Auto-generated method stub

		/**
		 * 读取训练数据集
		 */
		String iris_train = "H:/dataset\\iris\\iris.txt";
		
		String iris_test = "H:/dataset\\iris\\iris_test.txt";
		
		String[] labelSet = new String[] {"1","-1"};
		
		DataSet trainData = DataLoader.loalDataByTxt(iris_train, ",", 1, 1, 4, 2,labelSet);
		DataSet testData = DataLoader.loalDataByTxt(iris_test, ",", 1, 1, 4, 2,labelSet);
		
		System.out.println("train_data:"+JsonUtils.toJson(trainData));
	
		BPNetwork netWork = new BPNetwork(new CrossEntropyLoss());
		
		InputLayer inputLayer = new InputLayer(1,1,4);
		
		FullyLayer hidden1 = new FullyLayer(4, 40);
		
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(40, 20);
		
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(20, 2);

		SoftmaxWithCrossEntropyLayer hidden4 = new SoftmaxWithCrossEntropyLayer(2);
		
		netWork.addLayer(inputLayer);
		netWork.addLayer(hidden1);
		netWork.addLayer(active1);
		netWork.addLayer(hidden2);
		netWork.addLayer(active2);
		netWork.addLayer(hidden3);
		netWork.addLayer(hidden4);

//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 2000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 100000, 0.00001f, 10, LearnRateUpdate.NONE, false);
		
//		    netWork.GRADIENT_CHECK = true;
		
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	@Async
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
		
		DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, 1, 1 ,784, true);
		
		DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, 1, 1 ,784, true);

		BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.momentum);
		
		netWork.learnRate = 0.001f;
		int inputCount = (int) (Math.sqrt(794)+10);
		
		InputLayer inputLayer = new InputLayer(1,1,784);
		
		FullyLayer hidden1 = new FullyLayer(784, inputCount,false);
		
		BNLayer bn1 = new BNLayer();
		
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(inputCount, inputCount,false);
		
		BNLayer bn2 = new BNLayer();
		
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(inputCount, inputCount,false);
		
		BNLayer bn3 = new BNLayer();
		
		ReluLayer active3 = new ReluLayer();
		
		FullyLayer hidden4 = new FullyLayer(inputCount, 10);
		
		SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//		SoftmaxLayer softmax = new SoftmaxLayer(10);
		
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
		netWork.addLayer(softmax);
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 20000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.0001f, 64, LearnRateUpdate.NONE, false);

//			netWork.GRADIENT_CHECK = true;
		
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public void cnnNetwork_mnist_demo() {
		// TODO Auto-generated method stub
		/**
		 * 读取训练数据集
		 */
		String mnist_train_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-images.idx3-ubyte";
		
		String mnist_train_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\train-labels.idx1-ubyte";
		
		String mnist_test_data = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-images.idx3-ubyte";
		
		String mnist_test_label = "C:\\Users\\Administrator\\Desktop\\dataset\\mnist\\t10k-labels.idx1-ubyte";
		
		String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
		
		DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, 1, 1 ,784,true);
		
		DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, 1, 1 ,784,true);

		int channel = 1;
		
		int height = 28;
		
		int width = 28;
		
		CNN netWork = new CNN(new CrossEntropyLoss(), UpdaterType.momentum);
		
		netWork.learnRate = 0.1f;
		
		InputLayer inputLayer = new InputLayer(channel, 1, 784);
		
		ConvolutionLayer conv1 = new ConvolutionLayer(channel, 6, width, height, 5, 5, 2, 1);
		
		BNLayer bn1 = new BNLayer();
				
		ReluLayer active1 = new ReluLayer();
		
		PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
		
		ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 12, pool1.oWidth, pool1.oHeight, 5, 5, 0, 1);
		
		BNLayer bn2 = new BNLayer();
		
		ReluLayer active2 = new ReluLayer();
		
		PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

		int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
		
		int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
		
		FullyLayer full1 = new FullyLayer(fInputCount, inputCount,false);
		
		BNLayer bn3 = new BNLayer();
		
		ReluLayer active3 = new ReluLayer();
		
		FullyLayer full2 = new FullyLayer(inputCount, 10);

		SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

		netWork.addLayer(inputLayer);
		netWork.addLayer(conv1);
		netWork.addLayer(bn1);
		netWork.addLayer(active1);
		netWork.addLayer(pool1);
		netWork.addLayer(conv2);
		netWork.addLayer(bn2);
		netWork.addLayer(active2);
		netWork.addLayer(pool2);
		netWork.addLayer(full1);
		netWork.addLayer(bn3);
		netWork.addLayer(active3);
		netWork.addLayer(full2);
		netWork.addLayer(softmax);
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 20000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);
		
//		netWork.GRADIENT_CHECK = true;
		
		try {

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1, 0.001f, 32, LearnRateUpdate.NONE, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
			NetworkInit network = netWork.save();
			
			System.out.println(JsonUtils.toJson(network));
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	@Async
	public void cnnNetwork_mnist() {
		// TODO Auto-generated method stub
		
		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 6, width, height, 5, 5, 2, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			LeakyReluLayer active1 = new LeakyReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 12, pool1.oWidth, pool1.oHeight, 5, 5, 0, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			LeakyReluLayer active2 = new LeakyReluLayer();
			
			DropoutLayer drop1 = new DropoutLayer(0.5f);
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn3 = new BNLayer();
			
			LeakyReluLayer active3 = new LeakyReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, 10);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(drop1);
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(full2);
			netWork.addLayer(softmax);

//			netWork.GRADIENT_CHECK = true;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.0001f, 128, LearnRateUpdate.NONE, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
//			
//			dataBase.getNetworks().put("cnnMnist", netWork);
//			
//			NetworkInit network = netWork.save();
//			
//			System.out.println(JsonUtils.toJson(network));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	@Override
	public void cnnNetwork_cifar10() {
		// TODO Auto-generated method stub
		
		try {

	    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, true, labelSet);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, true, labelSet);
			
			System.out.println("data is ready.");
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			netWork.addLayer(inputLayer);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 16, width, height, 3, 3, 1, 1, false);
			netWork.addLayer(conv1);
			
			BNLayer bn1 = new BNLayer();
			netWork.addLayer(bn1);
			
			ReluLayer active1 = new ReluLayer();
			netWork.addLayer(active1);

			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool1);
			
			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 32, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1,false);
			netWork.addLayer(conv3);
			
			BNLayer bn3 = new BNLayer();
			netWork.addLayer(bn3);
			
			ReluLayer active3 = new ReluLayer();
			netWork.addLayer(active3);
			
			PoolingLayer pool2 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool2);

			
			ConvolutionLayer conv4 = new ConvolutionLayer(pool2.oChannel, 64, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1,false);
			netWork.addLayer(conv4);
			
			BNLayer bn4 = new BNLayer();
			netWork.addLayer(bn4);
			
			ReluLayer active4 = new ReluLayer();
			netWork.addLayer(active4);
			
			PoolingLayer pool3 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool3);


			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
//			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, 256, true);
			netWork.addLayer(full1);
			
//			BNLayer bn5 = new BNLayer();
//			netWork.addLayer(bn5);
			
			ReluLayer active9 = new ReluLayer();
			netWork.addLayer(active9);

			DropoutLayer drop1 = new DropoutLayer(0.5f);
			netWork.addLayer(drop1);

			FullyLayer full2 = new FullyLayer(full1.oWidth, 10, true);
			netWork.addLayer(full2);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
			netWork.addLayer(softmax);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.001f, 64, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	@Override
	public void cnnNetwork_vgg16_cifar10() {
		// TODO Auto-generated method stub
		
		try {

	    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			float[] mean = new float[] {0.485f, 0.456f, 0.406f};
			float[] std = new float[] {0.229f, 0.224f, 0.225f};
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			/**
			 * block1
			 */
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();

			PoolingLayer pool1 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			/**
			 * block2
			 */
			ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 128, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 128, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block3
			 */
			ConvolutionLayer conv5 = new ConvolutionLayer(pool2.oChannel, 256, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			ConvolutionLayer conv6 = new ConvolutionLayer(conv5.oChannel, 256, conv5.oWidth, conv5.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn6 = new BNLayer();
			
			ReluLayer active6 = new ReluLayer();
			
			ConvolutionLayer conv7 = new ConvolutionLayer(conv6.oChannel, 256, conv6.oWidth, conv6.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active7 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv7.oChannel, conv7.oWidth, conv7.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block4
			 */
			ConvolutionLayer conv8 = new ConvolutionLayer(pool3.oChannel, 512, pool3.oWidth, pool3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn8 = new BNLayer();
			
			ReluLayer active8 = new ReluLayer();
			
			ConvolutionLayer conv9 = new ConvolutionLayer(conv8.oChannel, 512, conv8.oWidth, conv8.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn9 = new BNLayer();
			
			ReluLayer active9 = new ReluLayer();
			
			ConvolutionLayer conv10 = new ConvolutionLayer(conv9.oChannel, 512, conv9.oWidth, conv9.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn10 = new BNLayer();
			
			ReluLayer active10 = new ReluLayer();

			PoolingLayer pool4 = new PoolingLayer(conv10.oChannel, conv10.oWidth, conv10.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block5
			 */
			ConvolutionLayer conv11 = new ConvolutionLayer(pool4.oChannel, 512, pool4.oWidth, pool4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn11 = new BNLayer();
			
			ReluLayer active11 = new ReluLayer();
			
			ConvolutionLayer conv12 = new ConvolutionLayer(conv11.oChannel, 512, conv11.oWidth, conv11.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn12 = new BNLayer();
			
			ReluLayer active12 = new ReluLayer();
			
			ConvolutionLayer conv13 = new ConvolutionLayer(conv12.oChannel, 512, conv12.oWidth, conv12.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn13 = new BNLayer();
			
			ReluLayer active13 = new ReluLayer();

			PoolingLayer pool5 = new PoolingLayer(conv13.oChannel, conv13.oWidth, conv13.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * fully
			 */

			int fInputCount = pool5.oChannel * pool5.oWidth * pool5.oHeight;
			
//			System.out.println(fInputCount);
			
			int inputCount = 256;
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn14 = new BNLayer();
			
			ReluLayer active14 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, inputCount, false);
			
			BNLayer bn15 = new BNLayer();
			
			ReluLayer active15 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(inputCount, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
			
			/**
			 * 装载网络
			 */
			netWork.addLayer(inputLayer);
			/**
			 * block1
			 */
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool1);
			/**
			 * block2
			 */
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(active4);
			netWork.addLayer(pool2);
			/**
			 * block3
			 */
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active5);
			netWork.addLayer(conv6);
			netWork.addLayer(bn6);
			netWork.addLayer(active6);
			netWork.addLayer(conv7);
			netWork.addLayer(bn7);
			netWork.addLayer(active7);
			netWork.addLayer(pool3);
			/**
			 * block4
			 */
			netWork.addLayer(conv8);
			netWork.addLayer(bn8);
			netWork.addLayer(active8);
			netWork.addLayer(conv9);
			netWork.addLayer(bn9);
			netWork.addLayer(active9);
			netWork.addLayer(conv10);
			netWork.addLayer(bn10);
			netWork.addLayer(active10);
			netWork.addLayer(pool4);
			/**
			 * block5
			 */
			netWork.addLayer(conv11);
			netWork.addLayer(bn11);
			netWork.addLayer(active11);
			netWork.addLayer(conv12);
			netWork.addLayer(bn12);
			netWork.addLayer(active12);
			netWork.addLayer(conv13);
			netWork.addLayer(bn13);
			netWork.addLayer(active13);
			netWork.addLayer(pool5);
			/**
			 * fully
			 */
			netWork.addLayer(full1);
			netWork.addLayer(bn14);
			netWork.addLayer(active14);
			netWork.addLayer(full2);
			netWork.addLayer(bn15);
			netWork.addLayer(active15);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	@Override
	public void showImage() {
		// TODO Auto-generated method stub
		String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    	
		String test_data_filename = "H:/dataset/cifar-10-binary.tar/cifar-10-binary/cifar-10-batches-bin/test_batch.bin";
		
		DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, false, labelSet);
    	
//		MatrixOperation.printImage(trainData.input.maxtir[0][0]);
		
		ImageUtils rc = new ImageUtils();
		
		String testOutPath = "H:/dataset\\r.png";

		System.out.println(testData.labels[10]);
		
		rc.createRGBImage(testOutPath, "png", testData.input.maxtir[10][1], 2);
		
	}

	@Override
	@Async
	public void alexNet_mnist() {
		// TODO Auto-generated method stub
		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.0001f;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 32, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 64, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool2.oChannel, 128, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 256, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 256, conv4.oWidth, conv4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv5.oChannel, conv5.oWidth, conv5.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 1024, false);

			BNLayer bn6 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(1024, 512, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(512, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//			SoftmaxLayer softmax = new SoftmaxLayer(10);
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool2);
			
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active3);
			
			netWork.addLayer(pool3);
			
			netWork.addLayer(full1);
			netWork.addLayer(bn6);
			netWork.addLayer(active4);
			netWork.addLayer(full2);
			netWork.addLayer(bn7);
			netWork.addLayer(active5);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.0001f, 128, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	@Override
	@Async
	public void alexNet_cifar10() {
		// TODO Auto-generated method stub
		
		try {

			String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			float[] mean = new float[] {0.485f, 0.456f, 0.406f};
			float[] std = new float[] {0.229f, 0.224f, 0.225f};
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 32, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 64, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool2.oChannel, 128, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 256, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 256, conv4.oWidth, conv4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv5.oChannel, conv5.oWidth, conv5.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 1024, false);

			BNLayer bn6 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(1024, 512, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(512, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//			SoftmaxLayer softmax = new SoftmaxLayer(10);
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool2);
			
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active3);
			
			netWork.addLayer(pool3);
			
			netWork.addLayer(full1);
			netWork.addLayer(bn6);
			netWork.addLayer(active4);
			netWork.addLayer(full2);
			netWork.addLayer(bn7);
			netWork.addLayer(active5);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.0001f, 128, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	

	@Override
	@Async
	public void vgg16_cifar10() {
		// TODO Auto-generated method stub

		try {

	    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			float[] mean = new float[] {0.5f,0.5f,0.5f};
			float[] std = new float[] {0.5f,0.5f,0.5f};
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			/**
			 * block1
			 */
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
			
			ReluLayer active1 = new ReluLayer();
			
			ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active2 = new ReluLayer();

			PoolingLayer pool1 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			/**
			 * block2
			 */
			ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 128, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active3 = new ReluLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 128, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active4 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block3
			 */
			ConvolutionLayer conv5 = new ConvolutionLayer(pool2.oChannel, 256, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active5 = new ReluLayer();
			
			ConvolutionLayer conv6 = new ConvolutionLayer(conv5.oChannel, 256, conv5.oWidth, conv5.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active6 = new ReluLayer();
			
			ConvolutionLayer conv7 = new ConvolutionLayer(conv6.oChannel, 256, conv6.oWidth, conv6.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active7 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv7.oChannel, conv7.oWidth, conv7.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block4
			 */
			ConvolutionLayer conv8 = new ConvolutionLayer(pool3.oChannel, 512, pool3.oWidth, pool3.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active8 = new ReluLayer();
			
			ConvolutionLayer conv9 = new ConvolutionLayer(conv8.oChannel, 512, conv8.oWidth, conv8.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active9 = new ReluLayer();
			
			ConvolutionLayer conv10 = new ConvolutionLayer(conv9.oChannel, 512, conv9.oWidth, conv9.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active10 = new ReluLayer();
			
			ConvolutionLayer conv20 = new ConvolutionLayer(conv10.oChannel, 512, conv10.oWidth, conv10.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active20 = new ReluLayer();
			
			PoolingLayer pool4 = new PoolingLayer(conv20.oChannel, conv20.oWidth, conv20.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block5
			 */
			ConvolutionLayer conv11 = new ConvolutionLayer(pool4.oChannel, 512, pool4.oWidth, pool4.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active11 = new ReluLayer();
			
			ConvolutionLayer conv12 = new ConvolutionLayer(conv11.oChannel, 512, conv11.oWidth, conv11.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active12 = new ReluLayer();
			
			ConvolutionLayer conv13 = new ConvolutionLayer(conv12.oChannel, 512, conv12.oWidth, conv12.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active13 = new ReluLayer();
			
			ConvolutionLayer conv21 = new ConvolutionLayer(conv13.oChannel, 512, conv13.oWidth, conv13.oHeight, 3, 3, 1, 1, false);
			
			ReluLayer active21 = new ReluLayer();

			PoolingLayer pool5 = new PoolingLayer(conv21.oChannel, conv21.oWidth, conv21.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * fully
			 */
			int fInputCount = pool5.oChannel * pool5.oWidth * pool5.oHeight;

			int inputCount = 4096;
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			ReluLayer active14 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, inputCount, false);
			
			ReluLayer active15 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(inputCount, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
			
			/**
			 * 装载网络
			 */
			netWork.addLayer(inputLayer);
			/**
			 * block1
			 */
			netWork.addLayer(conv1);
			netWork.addLayer(active1);
			netWork.addLayer(conv2);
			netWork.addLayer(active2);
			netWork.addLayer(pool1);
			/**
			 * block2
			 */
			netWork.addLayer(conv3);
			netWork.addLayer(active3);
			netWork.addLayer(conv4);
			netWork.addLayer(active4);
			netWork.addLayer(pool2);
			/**
			 * block3
			 */
			netWork.addLayer(conv5);
			netWork.addLayer(active5);
			netWork.addLayer(conv6);
			netWork.addLayer(active6);
			netWork.addLayer(conv7);
			netWork.addLayer(active7);
			netWork.addLayer(pool3);
			/**
			 * block4
			 */
			netWork.addLayer(conv8);
			netWork.addLayer(active8);
			netWork.addLayer(conv9);
			netWork.addLayer(active9);
			netWork.addLayer(conv10);
			netWork.addLayer(active10);
			netWork.addLayer(conv20);
			netWork.addLayer(active20);
			netWork.addLayer(pool4);
			/**
			 * block5
			 */
			netWork.addLayer(conv11);
			netWork.addLayer(active11);
			netWork.addLayer(conv12);
			netWork.addLayer(active12);
			netWork.addLayer(conv13);
			netWork.addLayer(active13);
			netWork.addLayer(conv21);
			netWork.addLayer(active21);
			netWork.addLayer(pool5);
			/**
			 * fully
			 */
			netWork.addLayer(full1);
			netWork.addLayer(active14);
			netWork.addLayer(full2);
			netWork.addLayer(active15);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	@Override
	public void bpNetwork_mnist(String sid, float lr) {
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

		BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.momentum);
		
		netWork.learnRate = lr;
		
		int inputCount = (int) (Math.sqrt(794)+10);
		
		InputLayer inputLayer = new InputLayer(1,1,784);
		
		FullyLayer hidden1 = new FullyLayer(784, inputCount,false);
		
		BNLayer bn1 = new BNLayer();
		
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(inputCount, inputCount,false);
		
		BNLayer bn2 = new BNLayer();
		
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(inputCount, inputCount,false);
		
		BNLayer bn3 = new BNLayer();
		
		ReluLayer active3 = new ReluLayer();
		
		FullyLayer hidden4 = new FullyLayer(inputCount, 10);
		
		SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//		SoftmaxLayer softmax = new SoftmaxLayer(10);
		
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
		netWork.addLayer(softmax);
		
//		SGDOptimizer optimizer = new SGDOptimizer(netWork, 20000, 0.001d);
		
//		BGDOptimizer optimizer = new BGDOptimizer(netWork, 20000, 0.001d);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(sid, netWork, 30, 0.0001f, 64, LearnRateUpdate.NONE, false);

//			netWork.GRADIENT_CHECK = true;
		
			TrainTask.addTask(sid, optimizer);
			
			optimizer.online(true);
			
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	@Override
	public void alexNet_mnist(String sid, float lr) {
		// TODO Auto-generated method stub
		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = lr;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 32, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 64, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool2.oChannel, 128, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 256, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 256, conv4.oWidth, conv4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv5.oChannel, conv5.oWidth, conv5.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 1024, false);

			BNLayer bn6 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(1024, 512, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(512, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//			SoftmaxLayer softmax = new SoftmaxLayer(10);
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool2);
			
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active3);
			
			netWork.addLayer(pool3);
			
			netWork.addLayer(full1);
			netWork.addLayer(bn6);
			netWork.addLayer(active4);
			netWork.addLayer(full2);
			netWork.addLayer(bn7);
			netWork.addLayer(active5);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(sid, netWork, 10, 0.0001f, 128, LearnRateUpdate.CONSTANT, false);

			TrainTask.addTask(sid, optimizer);
		
			optimizer.online(true);
			
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	@Override
	public void alexNet_cifar10(String sid, float lr) {
		// TODO Auto-generated method stub

		try {

			String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			float[] mean = new float[] {0.485f, 0.456f, 0.406f};
			float[] std = new float[] {0.229f, 0.224f, 0.225f};
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = lr;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 32, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 64, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool2.oChannel, 128, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 256, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 256, conv4.oWidth, conv4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv5.oChannel, conv5.oWidth, conv5.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 1024, false);

			BNLayer bn6 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(1024, 512, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(512, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

//			SoftmaxLayer softmax = new SoftmaxLayer(10);
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool2);
			
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active3);
			
			netWork.addLayer(pool3);
			
			netWork.addLayer(full1);
			netWork.addLayer(bn6);
			netWork.addLayer(active4);
			netWork.addLayer(full2);
			netWork.addLayer(bn7);
			netWork.addLayer(active5);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(sid, netWork, 20, 0.0001f, 128, LearnRateUpdate.CONSTANT, false);

			TrainTask.addTask(sid, optimizer);
			
			optimizer.online(true);
			
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	@Override
	public void cnnNetwork_vgg16_cifar10(String sid, float lr) {
		// TODO Auto-generated method stub
		try {

	    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			float[] mean = new float[] {0.485f, 0.456f, 0.406f};
			float[] std = new float[] {0.229f, 0.224f, 0.225f};
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = lr;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			/**
			 * block1
			 */
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();

			PoolingLayer pool1 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			/**
			 * block2
			 */
			ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 128, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn3 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 128, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn4 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block3
			 */
			ConvolutionLayer conv5 = new ConvolutionLayer(pool2.oChannel, 256, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn5 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			ConvolutionLayer conv6 = new ConvolutionLayer(conv5.oChannel, 256, conv5.oWidth, conv5.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn6 = new BNLayer();
			
			ReluLayer active6 = new ReluLayer();
			
			ConvolutionLayer conv7 = new ConvolutionLayer(conv6.oChannel, 256, conv6.oWidth, conv6.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn7 = new BNLayer();
			
			ReluLayer active7 = new ReluLayer();
			
			PoolingLayer pool3 = new PoolingLayer(conv7.oChannel, conv7.oWidth, conv7.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block4
			 */
			ConvolutionLayer conv8 = new ConvolutionLayer(pool3.oChannel, 512, pool3.oWidth, pool3.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn8 = new BNLayer();
			
			ReluLayer active8 = new ReluLayer();
			
			ConvolutionLayer conv9 = new ConvolutionLayer(conv8.oChannel, 512, conv8.oWidth, conv8.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn9 = new BNLayer();
			
			ReluLayer active9 = new ReluLayer();
			
			ConvolutionLayer conv10 = new ConvolutionLayer(conv9.oChannel, 512, conv9.oWidth, conv9.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn10 = new BNLayer();
			
			ReluLayer active10 = new ReluLayer();
			
			PoolingLayer pool4 = new PoolingLayer(conv10.oChannel, conv10.oWidth, conv10.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block5
			 */
			ConvolutionLayer conv11 = new ConvolutionLayer(pool4.oChannel, 512, pool4.oWidth, pool4.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn11 = new BNLayer();
			
			ReluLayer active11 = new ReluLayer();
			
			ConvolutionLayer conv12 = new ConvolutionLayer(conv11.oChannel, 512, conv11.oWidth, conv11.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn12 = new BNLayer();
			
			ReluLayer active12 = new ReluLayer();
			
			ConvolutionLayer conv13 = new ConvolutionLayer(conv12.oChannel, 512, conv12.oWidth, conv12.oHeight, 3, 3, 1, 1, false);
			
			BNLayer bn13 = new BNLayer();
			
			ReluLayer active13 = new ReluLayer();

			PoolingLayer pool5 = new PoolingLayer(conv13.oChannel, conv13.oWidth, conv13.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * fully
			 */

			int fInputCount = pool5.oChannel * pool5.oWidth * pool5.oHeight;

			int inputCount = 4096;
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn14 = new BNLayer();
			
			ReluLayer active14 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(full1.oWidth, 1000, false);
			
			BNLayer bn15 = new BNLayer();
			
			ReluLayer active15 = new ReluLayer();
			
			FullyLayer full3 = new FullyLayer(full2.oWidth, 10);

			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
			
			/**
			 * 装载网络
			 */
			netWork.addLayer(inputLayer);
			/**
			 * block1
			 */
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(pool1);
			/**
			 * block2
			 */
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(conv4);
			netWork.addLayer(bn4);
			netWork.addLayer(active4);
			netWork.addLayer(pool2);
			/**
			 * block3
			 */
			netWork.addLayer(conv5);
			netWork.addLayer(bn5);
			netWork.addLayer(active5);
			netWork.addLayer(conv6);
			netWork.addLayer(bn6);
			netWork.addLayer(active6);
			netWork.addLayer(conv7);
			netWork.addLayer(bn7);
			netWork.addLayer(active7);
			netWork.addLayer(pool3);
			/**
			 * block4
			 */
			netWork.addLayer(conv8);
			netWork.addLayer(bn8);
			netWork.addLayer(active8);
			netWork.addLayer(conv9);
			netWork.addLayer(bn9);
			netWork.addLayer(active9);
			netWork.addLayer(conv10);
			netWork.addLayer(bn10);
			netWork.addLayer(active10);
			netWork.addLayer(pool4);
			/**
			 * block5
			 */
			netWork.addLayer(conv11);
			netWork.addLayer(bn11);
			netWork.addLayer(active11);
			netWork.addLayer(conv12);
			netWork.addLayer(bn12);
			netWork.addLayer(active12);
			netWork.addLayer(conv13);
			netWork.addLayer(bn13);
			netWork.addLayer(active13);
			netWork.addLayer(pool5);
			/**
			 * fully
			 */
			netWork.addLayer(full1);
			netWork.addLayer(bn14);
			netWork.addLayer(active14);
			netWork.addLayer(full2);
			netWork.addLayer(bn15);
			netWork.addLayer(active15);
			netWork.addLayer(full3);
			netWork.addLayer(softmax);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(sid, netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);
			
			TrainTask.addTask(sid, optimizer);
			
			optimizer.online(true);
			
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}


	@Override
	public void cnnNetwork_mnist(String sid, float lr) {
		// TODO Auto-generated method stub

		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = lr;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 6, width, height, 5, 5, 2, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			LeakyReluLayer active1 = new LeakyReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 12, pool1.oWidth, pool1.oHeight, 5, 5, 0, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			LeakyReluLayer active2 = new LeakyReluLayer();
			
			DropoutLayer drop1 = new DropoutLayer(0.5f);
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn3 = new BNLayer();
			
			LeakyReluLayer active3 = new LeakyReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, 10);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(drop1);
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(full2);
			netWork.addLayer(softmax);

//			netWork.GRADIENT_CHECK = true;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(sid, netWork, 10, 0.0001f, 128, LearnRateUpdate.NONE, false);

			TrainTask.addTask(sid, optimizer);
			
			optimizer.online(true);
			
			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
//			
//			dataBase.getNetworks().put("cnnMnist", netWork);
//			
//			NetworkInit network = netWork.save();
//			
//			System.out.println(JsonUtils.toJson(network));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}


	@Override
	public void cnn_1x1() {
		// TODO Auto-generated method stub

		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 128, width, height, 5, 5, 2, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			// block start
			ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 1, 1, 0, 1, false);
			
			System.out.println(conv1.oWidth);
			
			ReluLayer active2 = new ReluLayer();
			
			ConvolutionLayer conv3 = new ConvolutionLayer(conv2.oChannel, 64, conv2.oWidth, conv2.oHeight, 3, 3, 1, 1, false);
			
			System.out.println(conv2.oWidth);
			
			ReluLayer active3 = new ReluLayer();
			
			ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 128, conv3.oWidth, conv3.oHeight, 1, 1, 0, 1, false);
			

			System.out.println(conv3.oWidth);
			System.out.println(conv4.oWidth);
			
			ReluLayer active4 = new ReluLayer();
			// block end
			
			PoolingLayer pool2 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn3 = new BNLayer();
			
			ReluLayer active5 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, 10);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			
			netWork.addLayer(conv2);
			netWork.addLayer(active2);
			netWork.addLayer(conv3);
			netWork.addLayer(active3);
			netWork.addLayer(conv4);
			netWork.addLayer(active4);
			
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
			netWork.addLayer(bn3);
			netWork.addLayer(active5);
			netWork.addLayer(full2);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.0001f, 128, LearnRateUpdate.NONE, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	@Override
	public void resnet18_mnist() {
		// TODO Auto-generated method stub

		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
			BNLayer bn1 = new BNLayer();
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block1  64 * 28 * 28
			 */
			BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, false);
			ReluLayer active2 = new ReluLayer();

			/**
			 * block2  64 * 28 * 28
			 */
			BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, false);
			ReluLayer active3 = new ReluLayer();
			
			/**
			 * block3  128 * 14 * 14
			 * downSample 28 / 2 = 14
			 */
			BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, true);
			ReluLayer active4 = new ReluLayer();

			/**
			 * block4  128 * 14 * 14
			 */
			BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, false);
			ReluLayer active5 = new ReluLayer();

			/**
			 * block5  256 * 7 * 7
			 * downSample 14 / 2 = 7
			 */
			BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, true);
			ReluLayer active6 = new ReluLayer();
			
			/**
			 * block6  256 * 7 * 7
			 */
			BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, false);
			ReluLayer active7 = new ReluLayer();

			/**
			 * block7  512 * 4 * 4
			 * downSample 7 / 2 = 4
			 */
			BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, true);
			ReluLayer active8 = new ReluLayer();
			
			
			/**
			 * block8  512 * 4 * 4
			 */
			BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, false);
			ReluLayer active9 = new ReluLayer();
			
			
			PoolingLayer pool2 = new PoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight, 2, 2, 2, PoolingType.MEAN_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, 10, false);
//
//			BNLayer bn5 = new BNLayer();
//			
//			ReluLayer active13 = new ReluLayer();
//			
//			FullyLayer full2 = new FullyLayer(inputCount, 10);
//			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			
			/**
			 * block1  64
			 */
			netWork.addLayer(bl1);
			netWork.addLayer(active2);
			netWork.addLayer(bl2);
			netWork.addLayer(active3);
			
			/**
			 * block2  128
			 */
			netWork.addLayer(bl3);
			netWork.addLayer(active4);
			netWork.addLayer(bl4);
			netWork.addLayer(active5);
			
			/**
			 * block3  256
			 */
			netWork.addLayer(bl5);
			netWork.addLayer(active6);
			netWork.addLayer(bl6);
			netWork.addLayer(active7);
			
			/**
			 * block4  512
			 */
			netWork.addLayer(bl7);
			netWork.addLayer(active8);
			netWork.addLayer(bl8);
			netWork.addLayer(active9);
			
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
//			netWork.addLayer(bn5);
//			netWork.addLayer(active13);
//			netWork.addLayer(full2);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.0001f, 128, LearnRateUpdate.NONE, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}


	@Override
	public void resnet18_cifar10() {
		// TODO Auto-generated method stub

		try {

			String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"H:/dataset/cifar-10/data_batch_1.bin",
					"H:/dataset/cifar-10/data_batch_2.bin",
					"H:/dataset/cifar-10/data_batch_3.bin",
					"H:/dataset/cifar-10/data_batch_4.bin",
					"H:/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, true, labelSet);
	    	
			System.out.println("data is ready.");
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001f;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1,false);
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
//			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block1  64 * 32 * 32
			 */
			BasicBlockLayer bl1 = new BasicBlockLayer(conv1.oChannel, 64, conv1.oHeight, conv1.oWidth, false);
			ReluLayer active2 = new ReluLayer();

			/**
			 * block2  64 * 32 * 32
			 */
			BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, false);
			ReluLayer active3 = new ReluLayer();
			
			/**
			 * block3  128 * 16 * 16
			 * downSample 32 / 2 = 16
			 */
			BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, true);
			ReluLayer active4 = new ReluLayer();

			/**
			 * block4  128 * 16 * 16
			 */
			BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, false);
			ReluLayer active5 = new ReluLayer();

			/**
			 * block5  256 * 8 * 8
			 * downSample 16 / 2 = 8
			 */
			BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, true);
			ReluLayer active6 = new ReluLayer();
			
			/**
			 * block6  256 * 8 * 8
			 */
			BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, false);
			ReluLayer active7 = new ReluLayer();

			/**
			 * block7  512 * 4 * 4
			 * downSample 8 / 2 = 4
			 */
			BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, true);
			ReluLayer active8 = new ReluLayer();
			
			
			/**
			 * block8  512 * 4 * 4
			 */
			BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, false);
			ReluLayer active9 = new ReluLayer();
			
			
			PoolingLayer pool2 = new PoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight, 2, 2, 2, PoolingType.MEAN_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, 10, false);
//
//			BNLayer bn5 = new BNLayer();
//			
//			ReluLayer active13 = new ReluLayer();
//			
//			FullyLayer full2 = new FullyLayer(inputCount, 10);
//			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
//			netWork.addLayer(pool1);
			
			/**
			 * block1  64
			 */
			netWork.addLayer(bl1);
			netWork.addLayer(active2);
			netWork.addLayer(bl2);
			netWork.addLayer(active3);
			
			/**
			 * block2  128
			 */
			netWork.addLayer(bl3);
			netWork.addLayer(active4);
			netWork.addLayer(bl4);
			netWork.addLayer(active5);
			
			/**
			 * block3  256
			 */
			netWork.addLayer(bl5);
			netWork.addLayer(active6);
			netWork.addLayer(bl6);
			netWork.addLayer(active7);
			
			/**
			 * block4  512
			 */
			netWork.addLayer(bl7);
			netWork.addLayer(active8);
			netWork.addLayer(bl8);
			netWork.addLayer(active9);
			
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
//			netWork.addLayer(bn5);
//			netWork.addLayer(active13);
//			netWork.addLayer(full2);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.0001f, 128, LearnRateUpdate.NONE, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, true, labelSet);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	
	public static void main(String[] args) {
		BusinessServiceImpl bs = new BusinessServiceImpl();
//		bs.showImage();
//		bs.bpNetwork_iris();
//		bs.bpNetwork_mnist();
//		bs.cnnNetwork_mnist_demo();
//		bs.cnnNetwork_mnist();
//		bs.cnnNetwork_cifar10();

//		bs.resnet18_cifar10();
		bs.resnet18_mnist();
//		bs.vgg16_cifar10();
//		bs.alexNet_mnist();
//		bs.alexNet_cifar10();
//		bs.cnnNetwork_vgg16_cifar10();
	}

}
