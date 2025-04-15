package com.omega.test.resnet.test;

import com.omega.common.tensor.Tensor;
import com.omega.common.utils.data.DataTransforms;
import com.omega.common.utils.data.DataLoader;
import com.omega.utils.image.ImageUtils;
import com.omega.utils.LabelUtils;
import com.omega.utils.MathUtils;
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

public class ResnetTest {
    public static void getImages() {
        try {
            int batchSize = 128;
            int channel = 3;
            int height = 32;
            int width = 32;
            String dpath = "H:/testImages/";
            String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
            String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
            String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
            DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, false);
            DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, false);
            Tensor input = new Tensor(batchSize, channel, height, width);
            Tensor label = new Tensor(batchSize, 1, 1, testData.labelSize);
            /**
             * 随机裁剪

             */
            DataTransforms.randomCrop(trainData.input, 32, 32, 4);
            /**
             * 随机翻转

             */
            DataTransforms.randomHorizontalFilp(trainData.input);
            ImageUtils iu = new ImageUtils();
            int[][] indexs = MathUtils.randomInts(testData.number, batchSize);
            /**
             * 遍历整个训练集

             */
            for (int it = 0; it < indexs.length; it++) {
                String filePath = dpath + it + "/";
                File file = new File(filePath);
                if (!file.exists()) {
                    file.mkdir();
                }
                testData.getRandomData(indexs[it], input, label);
                for (int n = 0; n < input.number; n++) {
                    String ol = LabelUtils.vectorTolabel(label.getByNumber(n), labelSet);
                    iu.createImage(n, input.getByNumber(n), ol, height, width, filePath, "png");
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            //	    	CUDAModules.initContext();
            ResnetTest resnet = new ResnetTest();
            //	    	resnet.resnet18_mnist();
            resnet.resnet18_cifar10();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void showImage() {
        // TODO Auto-generated method stub
        String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        String test_data_filename = "H:/dataset/cifar-10-binary.tar/cifar-10-binary/cifar-10-batches-bin/test_batch.bin";
        DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, false);
        //		MatrixOperation.printImage(trainData.input.maxtir[0][0]);
        ImageUtils rc = new ImageUtils();
        String testOutPath = "H:/dataset\\r.png";
        System.out.println(testData.labels[10]);
        rc.createRGBImage(testOutPath, "png", 32, 32, testData.input.getByNumberAndChannel(10, 0), 2);
    }

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
            String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
            File trainDataRes = new File(ResnetTest.class.getResource(mnist_train_data).toURI());
            File trainLabelRes = new File(ResnetTest.class.getResource(mnist_train_label).toURI());
            File testDataRes = new File(ResnetTest.class.getResource(mnist_test_data).toURI());
            File testLabelRes = new File(ResnetTest.class.getResource(mnist_test_label).toURI());
            DataSet trainData = DataLoader.loadDataByUByte(trainDataRes, trainLabelRes, labelSet, 1, 1, 784, true);
            DataSet testData = DataLoader.loadDataByUByte(testDataRes, testLabelRes, labelSet, 1, 1, 784, true);
            int channel = 1;
            int height = 28;
            int width = 28;
            CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
            netWork.learnRate = 0.001f;
            InputLayer inputLayer = new InputLayer(channel, 1, 784);
            ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
            BNLayer bn1 = new BNLayer();
            ReluLayer active1 = new ReluLayer();
            PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
            /**
             * block1  64 * 28 * 28

             */
            BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, 1, netWork);
            ReluLayer active2 = new ReluLayer();
            /**
             * block2  64 * 28 * 28

             */
            BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, 1, netWork);
            ReluLayer active3 = new ReluLayer();
            /**
             * block3  128 * 14 * 14
             * downSample 28 / 2 = 14

             */
            BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, 2, netWork);
            ReluLayer active4 = new ReluLayer();
            /**
             * block4  128 * 14 * 14

             */
            BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, 1, netWork);
            ReluLayer active5 = new ReluLayer();
            /**
             * block5  256 * 7 * 7
             * downSample 14 / 2 = 7

             */
            BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, 2, netWork);
            ReluLayer active6 = new ReluLayer();
            /**
             * block6  256 * 7 * 7

             */
            BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, 1, netWork);
            ReluLayer active7 = new ReluLayer();
            /**
             * block7  512 * 4 * 4
             * downSample 7 / 2 = 4

             */
            BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, 2, netWork);
            ReluLayer active8 = new ReluLayer();
            /**
             * block8  512 * 4 * 4

             */
            BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, 1, netWork);
            ReluLayer active9 = new ReluLayer();
            PoolingLayer pool2 = new PoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight, 4, 4, 4, PoolingType.MEAN_POOLING);
            int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
            FullyLayer full1 = new FullyLayer(fInputCount, 10, false);
            BNLayer bn5 = new BNLayer();
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
            netWork.addLayer(softmax);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.0001f, 128, LearnRateUpdate.NONE, false);
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

    public void resnet18_cifar10() {
        // TODO Auto-generated method stub
        try {
            String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
            String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
            String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
            float[] mean = new float[]{0.4914f, 0.4822f, 0.4465f};
            float[] std = new float[]{0.2023f, 0.1994f, 0.2010f};
            DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
            DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
            System.out.println("data is ready.");
            int channel = 3;
            int height = 32;
            int width = 32;
            int batchSize = 128;
            CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);
            netWork.CUDNN = true;
            netWork.learnRate = 0.01f;
            InputLayer inputLayer = new InputLayer(channel, height, width);
            ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
            conv1.paramsInit = ParamsInit.relu;
            BNLayer bn1 = new BNLayer();
            ReluLayer active1 = new ReluLayer();
            /**
             * block1  64 * 32 * 32

             */
            BasicBlockLayer bl1 = new BasicBlockLayer(conv1.oChannel, 64, conv1.oHeight, conv1.oWidth, 1, netWork);
            ReluLayer active2 = new ReluLayer();
            /**
             * block2  64 * 32 * 32

             */
            BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, 1, netWork);
            ReluLayer active3 = new ReluLayer();
            /**
             * block3  128 * 16 * 16
             * downSample 32 / 2 = 16

             */
            BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, 2, netWork);
            ReluLayer active4 = new ReluLayer();
            /**
             * block4  128 * 16 * 16

             */
            BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, 1, netWork);
            ReluLayer active5 = new ReluLayer();
            /**
             * block5  256 * 8 * 8
             * downSample 16 / 2 = 8

             */
            BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, 2, netWork);
            ReluLayer active6 = new ReluLayer();
            /**
             * block6  256 * 8 * 8

             */
            BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, 1, netWork);
            ReluLayer active7 = new ReluLayer();
            /**
             * block7  512 * 4 * 4
             * downSample 8 / 2 = 4

             */
            BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, 2, netWork);
            ReluLayer active8 = new ReluLayer();
            /**
             * block8  512 * 4 * 4

             */
            BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, 1, netWork);
            ReluLayer active9 = new ReluLayer();
            AVGPoolingLayer pool2 = new AVGPoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight);
            /**
             * fully  512 * 1 * 1

             */
            int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
            FullyLayer full1 = new FullyLayer(fInputCount, 10);
            SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
            netWork.addLayer(inputLayer);
            netWork.addLayer(conv1);
            netWork.addLayer(bn1);
            netWork.addLayer(active1);
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
            netWork.addLayer(softmax);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.0001f, batchSize, LearnRateUpdate.GD_GECAY, false);
            long start = System.currentTimeMillis();
            optimizer.train(trainData, testData, mean, std);
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

