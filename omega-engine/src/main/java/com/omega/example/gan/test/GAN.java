package com.omega.example.gan.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.GANOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.gan.utils.ImageDataLoader;

public class GAN {
    public static CNN NetG(int ngf, int nz) {
        CNN netWork = new CNN(LossType.BCE, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.0002f;
        InputLayer inputLayer = new InputLayer(nz, 1, 1);
        ConvolutionTransposeLayer convt1 = new ConvolutionTransposeLayer(nz, ngf * 8, 1, 1, 4, 4, 0, 1, 1, 0, false);
        BNLayer bn1 = new BNLayer();
        ReluLayer active1 = new ReluLayer();
        ConvolutionTransposeLayer convt2 = new ConvolutionTransposeLayer(convt1.oChannel, ngf * 4, convt1.oWidth, convt1.oHeight, 4, 4, 1, 2, 1, 0, false);
        BNLayer bn2 = new BNLayer();
        ReluLayer active2 = new ReluLayer();
        ConvolutionTransposeLayer convt3 = new ConvolutionTransposeLayer(convt2.oChannel, ngf * 2, convt2.oWidth, convt2.oHeight, 4, 4, 1, 2, 1, 0, false);
        BNLayer bn3 = new BNLayer();
        ReluLayer active3 = new ReluLayer();
        ConvolutionTransposeLayer convt4 = new ConvolutionTransposeLayer(convt3.oChannel, ngf, convt3.oWidth, convt3.oHeight, 4, 4, 1, 2, 1, 0, false);
        BNLayer bn4 = new BNLayer();
        ReluLayer active4 = new ReluLayer();
        ConvolutionTransposeLayer convt5 = new ConvolutionTransposeLayer(convt4.oChannel, 3, convt4.oWidth, convt4.oHeight, 5, 5, 1, 3, 1, 0, true);
        //		BNLayer bn5 = new BNLayer();
        TanhLayer active5 = new TanhLayer();
        //		SigmodLayer active5 = new SigmodLayer();
        netWork.addLayer(inputLayer);
        netWork.addLayer(convt1);
        netWork.addLayer(bn1);
        netWork.addLayer(active1);
        netWork.addLayer(convt2);
        netWork.addLayer(bn2);
        netWork.addLayer(active2);
        netWork.addLayer(convt3);
        netWork.addLayer(bn3);
        netWork.addLayer(active3);
        netWork.addLayer(convt4);
        netWork.addLayer(bn4);
        netWork.addLayer(active4);
        netWork.addLayer(convt5);
        //		netWork.addLayer(bn5);
        netWork.addLayer(active5);
        return netWork;
    }

    public static CNN NetD(int ndf, int imw, int imh) {
        CNN netWork = new CNN(LossType.BCE, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.0002f;
        netWork.PROPAGATE_DOWN = true;
        InputLayer inputLayer = new InputLayer(3, imh, imw);
        ConvolutionLayer conv1 = new ConvolutionLayer(3, ndf, imw, imh, 5, 5, 1, 3, false);
        BNLayer bn1 = new BNLayer();
        LeakyReluLayer active1 = new LeakyReluLayer();
        ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, ndf * 2, conv1.oWidth, conv1.oHeight, 4, 4, 1, 2, false);
        BNLayer bn2 = new BNLayer();
        LeakyReluLayer active2 = new LeakyReluLayer();
        ConvolutionLayer conv3 = new ConvolutionLayer(conv2.oChannel, ndf * 4, conv2.oWidth, conv2.oHeight, 4, 4, 1, 2, false);
        BNLayer bn3 = new BNLayer();
        LeakyReluLayer active3 = new LeakyReluLayer();
        ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, ndf * 8, conv3.oWidth, conv3.oHeight, 4, 4, 1, 2, false);
        BNLayer bn4 = new BNLayer();
        LeakyReluLayer active4 = new LeakyReluLayer();
        ConvolutionLayer conv5 = new ConvolutionLayer(conv4.oChannel, 1, conv4.oWidth, conv4.oHeight, 4, 4, 0, 1, true);
        SigmodLayer active5 = new SigmodLayer();
        netWork.addLayer(inputLayer);
        netWork.addLayer(conv1);
        netWork.addLayer(bn1);
        netWork.addLayer(active1);
        netWork.addLayer(conv2);
        netWork.addLayer(bn2);
        netWork.addLayer(active2);
        netWork.addLayer(conv3);
        netWork.addLayer(bn3);
        netWork.addLayer(active3);
        netWork.addLayer(conv4);
        netWork.addLayer(bn4);
        netWork.addLayer(active4);
        netWork.addLayer(conv5);
        netWork.addLayer(active5);
        return netWork;
    }

    public static void gan_anime() {
        int imw = 96;
        int imh = 96;
        int ngf = 64; //生成器featrue map数
        int ndf = 64; //判别器feature map数
        int nz = 100; //噪声维度
        int batchSize = 512;
        int d_every = 3;
        int g_every = 1;
        try {
            String imgDirPath = "H://voc//gan_anime//faces//";
            CNN netG = NetG(ngf, nz);
            CNN netD = NetD(ndf, imw, imh);
            ImageDataLoader dataLoader = new ImageDataLoader(imgDirPath, imw, imh, batchSize);
            GANOptimizer optimizer = new GANOptimizer(netG, netD, batchSize, 2000, d_every, g_every, 0.001f, LearnRateUpdate.SMART_HALF, false);
            optimizer.train(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        try {
            //			CUDAModules.initContext();
            gan_anime();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

