package com.omega.example.resnet.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.LinkedHashMap;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.AVGPoolingLayer;
import com.omega.engine.nn.layer.BasicBlockLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;

/**
 * resnet18
 * @author Administrator
 *
 */
public class Resnet18 {
	
	public static CNN instance(int channel,int height,int width,int output) {
		
		CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);

		netWork.CUDNN = true;
		
		netWork.learnRate = 0.1f;
		
		InputLayer inputLayer = new InputLayer(channel, height, width);
		
		ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
		
		BNLayer bn1 = new BNLayer();
		
		ReluLayer active1 = new ReluLayer();
		
		PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
		
		/**
		 * block1  64 * 32 * 32
		 */
		BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, 1, netWork);
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
		
		FullyLayer full1 = new FullyLayer(fInputCount, output);

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
		
		return netWork;
	}
	
	public static void loadWeight(CNN network,String path) {
		
		try {
			
			File file = new File(path);
			
			if(file.exists()) {
				
				FileReader fileReader = new FileReader(file);
		        Reader reader = new InputStreamReader(new FileInputStream(file), "Utf-8");
		        int ch= 0;
		        StringBuffer sb = new StringBuffer();
		        while((ch = reader.read()) != -1) {
		           sb.append((char) ch);
		        }
		        fileReader.close();
		        reader.close();
		        String jsonStr = sb.toString();
				
		        System.out.println(jsonStr.getBytes().length / 1024 / 1024 + "m");
		        
		        Map<String,Object> layers = new LinkedHashMap<String, Object>();
		        
		        layers = JsonUtils.gson.fromJson(jsonStr, layers.getClass());

			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	
	public static void main(String args[]) {
		
		String path = "H:\\voc\\train\\resnet18.json";
		
		try {
			
//			CUDAModules.initContext();
			
			CNN resnet = Resnet18.instance(3, 224, 224, 1000);
			
			Resnet18.loadWeight(resnet, path);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {
			CUDAMemoryManager.free();
		}
		
	}
	
	
}
