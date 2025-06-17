package com.omega.example.alexnet.test;

import com.omega.common.utils.DataLoader;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
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

public class AlexnetTest {
    public static void main(String[] args) {
        try {
            //	    	CUDAModules.initContext();
            AlexnetTest alexnet = new AlexnetTest();
            //	    	alexnet.alexNet_mnist();
            alexnet.alexNet_cifar10();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

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
            String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
            File trainDataRes = new File(AlexnetTest.class.getResource(mnist_train_data).toURI());
            File trainLabelRes = new File(AlexnetTest.class.getResource(mnist_train_label).toURI());
            File testDataRes = new File(AlexnetTest.class.getResource(mnist_test_data).toURI());
            File testLabelRes = new File(AlexnetTest.class.getResource(mnist_test_label).toURI());
            DataSet trainData = DataLoader.loadDataByUByte(trainDataRes, trainLabelRes, labelSet, 1, 1, 784, true);
            DataSet testData = DataLoader.loadDataByUByte(testDataRes, testLabelRes, labelSet, 1, 1, 784, true);
            int channel = 1;
            int height = 28;
            int width = 28;
            CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.1f;
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
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.0001f, 128, LearnRateUpdate.GD_GECAY, false);
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

    public void alexNet_cifar10() {
        // TODO Auto-generated method stub
        try {
            String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
            String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
            String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
            float[] mean = new float[]{0.485f, 0.456f, 0.406f};
            float[] std = new float[]{0.229f, 0.224f, 0.225f};
            DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
            DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
            System.out.println("data is ready.");
            int channel = 3;
            int height = 32;
            int width = 32;
            CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.001f;
            InputLayer inputLayer = new InputLayer(channel, height, width);
            ConvolutionLayer conv1 = new ConvolutionLayer(channel, 96, width, height, 3, 3, 1, 1, false);
            BNLayer bn1 = new BNLayer();
            ReluLayer active1 = new ReluLayer();
            PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 256, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
            BNLayer bn2 = new BNLayer();
            ReluLayer active2 = new ReluLayer();
            PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            ConvolutionLayer conv3 = new ConvolutionLayer(pool2.oChannel, 384, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
            BNLayer bn3 = new BNLayer();
            ReluLayer active3 = new ReluLayer();
            ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 384, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
            BNLayer bn4 = new BNLayer();
            ReluLayer active4 = new ReluLayer();
            ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 256, conv4.oWidth, conv4.oHeight, 3, 3, 1, 1, false);
            BNLayer bn5 = new BNLayer();
            ReluLayer active5 = new ReluLayer();
            PoolingLayer pool3 = new PoolingLayer(conv5.oChannel, conv5.oWidth, conv5.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
            FullyLayer full1 = new FullyLayer(fInputCount, 4096, false);
            BNLayer bn6 = new BNLayer();
            ReluLayer active6 = new ReluLayer();
            FullyLayer full2 = new FullyLayer(4096, 4096, false);
            BNLayer bn7 = new BNLayer();
            ReluLayer active7 = new ReluLayer();
            FullyLayer full3 = new FullyLayer(4096, 10);
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
            netWork.addLayer(conv3);
            netWork.addLayer(bn3);
            netWork.addLayer(active3);
            netWork.addLayer(conv4);
            netWork.addLayer(bn4);
            netWork.addLayer(active4);
            netWork.addLayer(conv5);
            netWork.addLayer(bn5);
            netWork.addLayer(active5);
            netWork.addLayer(pool3);
            netWork.addLayer(full1);
            netWork.addLayer(bn6);
            netWork.addLayer(active6);
            netWork.addLayer(full2);
            netWork.addLayer(bn7);
            netWork.addLayer(active7);
            netWork.addLayer(full3);
            netWork.addLayer(softmax);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 50, 0.0001f, 128, LearnRateUpdate.GD_GECAY, false);
            long start = System.currentTimeMillis();
            optimizer.train(trainData, testData, mean, std);
            optimizer.test(testData);
            System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
            //			NetworkUtils.save(netWork, "H://test2.json", "test2");
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

