package com.omega.example.gan.test;

import java.io.File;

import com.omega.common.utils.DataLoader;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.optimizer.GANOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;

public class MinistGAN {
	
	public static BPNetwork NetG(int imgSize,int latentSize) {
		
		BPNetwork netWork = new BPNetwork(LossType.MSE, UpdaterType.adamw);

		netWork.CUDNN = true;
		
		netWork.learnRate = 0.0001f;
		
		InputLayer inputLayer = new InputLayer(1, 1, latentSize);
		
		FullyLayer full1 = new FullyLayer(latentSize, 256, true);
		ReluLayer active1 = new ReluLayer();
		FullyLayer full2 = new FullyLayer(256, 256, true);
		ReluLayer active2 = new ReluLayer();
		FullyLayer full3 = new FullyLayer(256, imgSize, true);
		TanhLayer active4 = new TanhLayer();

		netWork.addLayer(inputLayer);
		netWork.addLayer(full1);
		netWork.addLayer(active1);
		netWork.addLayer(full2);
		netWork.addLayer(active2);
		netWork.addLayer(full3);
		netWork.addLayer(active4);
		return netWork;
	}
	
	public BPNetwork NetD(int imgSize) {
		
		BPNetwork netWork = new BPNetwork(LossType.MSE, UpdaterType.adamw);

		netWork.CUDNN = true;
		
		netWork.learnRate = 0.0001f;
		
		netWork.PROPAGATE_DOWN = true;
		
		InputLayer inputLayer = new InputLayer(1, 1, imgSize);
		
		FullyLayer full1 = new FullyLayer(imgSize, 256, true);
		LeakyReluLayer active1 = new LeakyReluLayer();
		FullyLayer full2 = new FullyLayer(256, 256, true);
		LeakyReluLayer active2 = new LeakyReluLayer();
		FullyLayer full3 = new FullyLayer(256, 1, true);
		SigmodLayer active4 = new SigmodLayer();
		
		netWork.addLayer(inputLayer);
		netWork.addLayer(full1);
		netWork.addLayer(active1);
		netWork.addLayer(full2);
		netWork.addLayer(active2);
		netWork.addLayer(full3);
		netWork.addLayer(active4);
		return netWork;
	}
	
	public void gan_anime() {
		
		int imgSize = 784;
		int ngf = 784; //生成器featrue map数
		int nz = 100; //噪声维度
		int batchSize = 2048;
		
		int d_every = 1;
		int g_every = 1;
		
		float[] mean = new float[] {0.5f};
		float[] std = new float[] {0.5f};
		
		try {
			
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			File trainDataRes = new File(this.getClass().getClassLoader().getResource(mnist_train_data).toURI());
			
			File trainLabelRes = new File(this.getClass().getClassLoader().getResource(mnist_train_label).toURI());

			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes, trainLabelRes, labelSet, 1, 1 , 784, true, mean, std);
			
			BPNetwork netG = NetG(ngf, nz);
			
			BPNetwork netD = NetD(imgSize);
			
			GANOptimizer optimizer = new GANOptimizer(netG, netD, batchSize, 3500, d_every, g_every, 0.001f, LearnRateUpdate.CONSTANT, false);
			
			optimizer.train(trainData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static void main(String args[]) {
		
		try {
			
//			CUDAModules.initContext();
			
			MinistGAN gan = new MinistGAN();
			
			gan.gan_anime();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	} 
	
}
