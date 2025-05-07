package com.omega.example.vggnet.test;

import com.omega.common.utils.DataLoader;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;

public class vggnetTest {
    public static void main(String[] args) {
        try {
            //	    	CUDAModules.initContext();
            vggnetTest vggnet = new vggnetTest();
            vggnet.vgg16_cifar10();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void vgg16_cifar10() {
        // TODO Auto-generated method stub
        try {
            String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
            String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
            String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
            float[] mean = new float[]{0.485f, 0.456f, 0.406f};
            float[] std = new float[]{0.229f, 0.224f, 0.225f};
            DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
            DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
            int channel = 3;
            int height = 32;
            int width = 32;
            CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adam);
            netWork.CUDNN = true;
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

