package com.omega.example.resnet.test;

import java.io.File;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;

import com.omega.engine.check.VailCodeCheck;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.resnet.data.VailCodeDataLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

public class VailCode {
	
	public static void resize() {

		try {
			
			int rsize = 160;
			
			String trainPath = "H:\\voc\\vailCode\\train_org1";

			String outPath = "H:\\voc\\vailCode\\train\\";
			
			File file = new File(trainPath);
			
			if(file.exists() && file.isDirectory()) {
				
				for(File img:file.listFiles()) {
					System.out.println(img.getName());
					String imgOutPath = outPath + "\\" + img.getName();
					YoloImageUtils.resize(img.getPath(), imgOutPath, rsize, rsize);
				}
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static void formatLabel() {

		try {
			
			String trainPath = "H:\\voc\\vailCode\\eval";

			String outPath = "H:\\voc\\vailCode\\eval.txt";
			
			Map<String,String> dataList = new HashMap<String, String>();
			
			File file = new File(trainPath);
			
			if(file.exists() && file.isDirectory()) {
				
				for(File img:file.listFiles()) {
					String filename = img.getName();
					String label = filename.split("_")[0];
					dataList.put(filename, label);
				}
				
			}
			
			File txt = new File(outPath);
			
			if(!txt.exists()) {
				txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
			}
			
			try (FileOutputStream fos = new FileOutputStream(txt);
//					OutputStreamWriter osr = new OutputStreamWriter(fos,"utf-8");
//					BufferedWriter bufferedWriter = new BufferedWriter(osr);
					) {
	 
				for (String name : dataList.keySet()) {
					String text = name;
					text += " " + dataList.get(name);
					text += "\n";
//					System.out.println(text);
					fos.write(text.getBytes());
				}
	 
				fos.flush();
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static void vailCode() {
		
		String[] labelSet = new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9","A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
	            "V", "W", "X", "Y", "Z","a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
	            "v", "w", "x", "y", "z"};
		try {
			
			int batchSize = 64;
			
			int labelClassLength = 4;
			
			String cfg_path = "H:\\voc\\vailCode\\vailCodeModel.cfg";
			
			String trainPath = "H:\\voc\\vailCode\\train";
			String trainLabelPath = "H:\\voc\\vailCode\\train.txt";
			
			String vailPath = "H:\\voc\\vailCode\\eval";
			String vailLabelPath = "H:\\voc\\vailCode\\eval.txt";
			
			VailCodeDataLoader trainDataLoader = new VailCodeDataLoader(trainPath, trainLabelPath, batchSize, labelSet, labelSet.length, labelClassLength);
			
			VailCodeDataLoader vailDataLoader = new VailCodeDataLoader(vailPath, vailLabelPath, batchSize, labelSet, labelSet.length, labelClassLength);
			
			VailCodeCheck check = new VailCodeCheck();
			
			CNN netWork = new CNN(LossType.multiLabel_soft_margin, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.0001f;

			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 100, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
			
			optimizer.train(trainDataLoader, vailDataLoader, check);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String args[]) {
		
		try {
			
//			CUDAModules.initContext();
			
//			resize();
			
//			formatLabel();
			
			vailCode();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
