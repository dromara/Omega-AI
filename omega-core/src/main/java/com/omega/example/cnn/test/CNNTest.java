package com.omega.example.cnn.test;

import com.omega.common.utils.DataLoader;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;

import java.io.File;

public class CNNTest {
    public static void main(String[] args) {
        try {
            //	    	CUDAModules.initContext();
            CNNTest cnn = new CNNTest();
            	    	cnn.cnnNetwork_mnist();
//            cnn.cnnNetwork_cifar10();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

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
            String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
            File trainDataRes = new File(CNNTest.class.getResource(mnist_train_data).toURI());
            File trainLabelRes = new File(CNNTest.class.getResource(mnist_train_label).toURI());
            File testDataRes = new File(CNNTest.class.getResource(mnist_test_data).toURI());
            File testLabelRes = new File(CNNTest.class.getResource(mnist_test_label).toURI());
            DataSet trainData = DataLoader.loadDataByUByte(trainDataRes, trainLabelRes, labelSet, 1, 1, 784, true);
            DataSet testData = DataLoader.loadDataByUByte(testDataRes, testLabelRes, labelSet, 1, 1, 784, true);
            int channel = 1;
            int height = 28;
            int width = 28;
            CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
            netWork.learnRate = 0.001f;
            InputLayer inputLayer = new InputLayer(channel, 1, 784);
            ConvolutionLayer conv1 = new ConvolutionLayer(channel, 6, width, height, 5, 5, 2, 1, false);
            BNLayer bn1 = new BNLayer();
            ReluLayer active1 = new ReluLayer();
            PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 12, pool1.oWidth, pool1.oHeight, 5, 5, 0, 1, false);
            BNLayer bn2 = new BNLayer();
            ReluLayer active2 = new ReluLayer();
            DropoutLayer drop1 = new DropoutLayer(0.5f);
            PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
            int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
            FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);
            BNLayer bn3 = new BNLayer();
            ReluLayer active3 = new ReluLayer();
            FullyLayer full2 = new FullyLayer(inputCount, 10);
            SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
            netWork.addLayer(inputLayer);
            netWork.addLayer(conv1);
            //			netWork.addLayer(bn1);
            netWork.addLayer(active1);
            netWork.addLayer(pool1);
            netWork.addLayer(conv2);
            //			netWork.addLayer(bn2);
            netWork.addLayer(active2);
            //			netWork.addLayer(drop1);
            netWork.addLayer(pool2);
            netWork.addLayer(full1);
            //			netWork.addLayer(bn3);
            netWork.addLayer(active3);
            netWork.addLayer(full2);
            netWork.addLayer(softmax);
            //			netWork.GRADIENT_CHECK = true;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.0001f, 128, LearnRateUpdate.NONE, false);
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
        } finally {
            try {
                CUDAMemoryManager.freeAll();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public void cnnNetwork_cifar10() {
        // TODO Auto-generated method stub
        try {
            String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
            String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
            String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
            DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
            DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true);
            System.out.println("data is ready.");
            int channel = 3;
            int height = 32;
            int width = 32;
            CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adam);
            netWork.learnRate = 0.01f;
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
            ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 32, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
            netWork.addLayer(conv3);
            BNLayer bn3 = new BNLayer();
            netWork.addLayer(bn3);
            ReluLayer active3 = new ReluLayer();
            netWork.addLayer(active3);
            PoolingLayer pool2 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            netWork.addLayer(pool2);
            ConvolutionLayer conv4 = new ConvolutionLayer(pool2.oChannel, 64, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
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
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);
            long start = System.currentTimeMillis();
            optimizer.train(trainData);
            optimizer.test(testData);
            System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            try {
                CUDAMemoryManager.freeAll();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }
}

