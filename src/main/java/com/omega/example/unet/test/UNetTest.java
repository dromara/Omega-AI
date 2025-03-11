package com.omega.example.unet.test;

import java.io.File;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.SimpleUNet;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.unet.utils.SegImageDataLoader;

public class UNetTest {
	
	
	public static void unet_cell() {
		
		try {
			
			boolean bias = false;
			
			int width = 256;
			int height = 256;
			
			int batchSize = 8;
			
			String imgDirPath = "H:\\unet-dataset\\dsb2018_256\\train\\";
			String vailImgDirPath = "H:\\unet-dataset\\dsb2018_256\\vail\\";
			String maskDirPath = "H:\\unet-dataset\\dsb2018_256\\masks\\";
			
			SegImageDataLoader dataLoader = new SegImageDataLoader(imgDirPath, maskDirPath, width, height, batchSize);
			
			SimpleUNet network = new SimpleUNet(LossType.BCE, UpdaterType.adamw, 3, 1, width, height, bias);
			network.CUDNN = true;
			network.learnRate = 0.001f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 50, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
			
			optimizer.trainSeg(dataLoader);
			
			network.RUN_MODEL = RunModel.TEST;
			
			File file = new File(vailImgDirPath);
			
			Tensor input = new Tensor(1, 3, height, width, true);
			
			Tensor label = new Tensor(1, 1, height, width, true);
			
			if(file.exists() && file.isDirectory()) {
				String[] filenames = file.list();
				for(int i = 0;i < filenames.length;i++) {
					String filename = filenames[i].split("\\.")[0];
					String extName = filenames[i].split("\\.")[1];
					testCell(vailImgDirPath, maskDirPath, filename, extName, dataLoader, height, width, network, input, label);
				}
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static void testCell(String path,String labelPath,String filename,String extName,SegImageDataLoader dataLoader,int height,int width,SimpleUNet network,Tensor input,Tensor label) {

		String testImg = path + filename + "." + extName;
		
		String testLabelImg = labelPath + filename + "." + extName;
		System.out.println(testImg);
		dataLoader.loadData(testImg, input);
		
		dataLoader.loadLabelData(testLabelImg, label);
		
		Tensor output = network.forward(input);
		
		ImageUtils utils = new ImageUtils();
		
		output.syncHost();
//	    PrintUtils.printImage(output);
	    for(int i = 0;i<output.dataLength;i++) {
	    	output.data[i] = output.data[i] * 255.0f;
		}

		String outImg = "H:\\unet-dataset\\cell\\" + filename + "." + extName;

		utils.createRGBImage(outImg, "png", ImageUtils.color2rgb2(output.data, output.channel, output.height, output.width, false), output.height, output.width, null, null);
		
		label.syncHost();
		for(int i = 0;i<label.dataLength;i++) {
			label.data[i] = label.data[i] * 255.0f;
		}

		String labelImg = "H:\\unet-dataset\\cell\\" + filename + "-mask" + "." + extName;
		
		utils.createRGBImage(labelImg, "png", ImageUtils.color2rgb2(label.data, label.channel, label.height, label.width, false), label.height, label.width, null, null);

	}
	
	public static void main(String[] args) {
		
		try {

//			CUDAModules.initContext();
			
			unet_cell();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
}
