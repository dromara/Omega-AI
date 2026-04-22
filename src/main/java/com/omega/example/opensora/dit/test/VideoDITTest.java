package com.omega.example.opensora.dit.test;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.video.OmegaVideo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BinDataType;

public class VideoDITTest {
	
	public static void complate_ms() throws Exception {

		String dataPath = "D:\\dataset\\video\\latend.bin";
        String clipDataPath = "D:\\dataset\\video\\clip.bin";

        String meanStdPath = "D:\\dataset\\video\\mean_std.bin";
        
        int batchSize = 1000;
        int latendDim = 128;
        int num_frames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * num_frames, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int[][] indexs = dataLoader.shuffle();

        Tensor latend = new Tensor(batchSize, latendDim, num_frames * dataLoader.height, dataLoader.width, true);
        Tensor condInput = new Tensor(batchSize * dataLoader.clipMaxTime, 1, 1, dataLoader.clipEmbd, true);
        
        int count = 10000;
        
        int tmp = count * num_frames * height * width - 1;
        
        float[] mean = RandomUtils.val(latendDim, 0.0f);
        float[] std = RandomUtils.val(latendDim, 0.0f);

        /**
         * 遍历整个训练集
         */
        for (int it = 0; it < 10; it++) {
            System.out.println("mean:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
//            	System.out.println(latend.height + ":" + latend.channel);
            	mean[c] += latend.data[i] / (tmp + 1);
            }

        }
        System.out.println("mean finish.");
        System.out.println(JsonUtils.toJson(mean));
        for (int it = 0; it < 10; it++) {
            System.out.println("std:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
            	std[c] += Math.pow(latend.data[i] - mean[c], 2) / tmp;
            }

        }
        
        for(int c = 0;c<latendDim;c++) {
        	std[c] = (float) Math.sqrt(std[c]);
        }
        
        System.out.println(JsonUtils.toJson(mean));
        System.out.println(JsonUtils.toJson(std));
        
        Tensor mean_t = new Tensor(latendDim, 1, 1, 1, mean, true);
        Tensor std_t = new Tensor(latendDim, 1, 1, 1, std, true);
        
        File file = new File(meanStdPath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
        	com.omega.engine.nn.network.utils.ModelUtils.saveParams(rFile, mean_t);
        	com.omega.engine.nn.network.utils.ModelUtils.saveParams(rFile, std_t);
            System.out.println("save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }

	}
	
	public static void video_dit_train() throws Exception {
		
		String dataPath = "D:\\dataset\\video\\latend.bin";
        String clipDataPath = "D:\\dataset\\video\\clip.bin";
        String meanStdPath = "D:\\dataset\\video\\mean_std.bin";
        
        int batchSize = 4;
        int latendDim = 128;
        int numFrames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * numFrames, height, width, maxContext, textEmbedDim, BinDataType.float32);
		
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        int maxContextLen = 77;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaVideo dit = new OmegaVideo(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, height, width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
         
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        
        loadMS(meanStdPath, mean, std);

        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 20, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_video_dit_ICPlan(dataLoader, icplan, "D:\\dataset\\video\\models\\video_dit_b_", mean, std, 2);
        
        String save_model_path = "D:\\dataset\\video\\models\\video_dit_b.model";
        ModelUtils.saveModel(dit, save_model_path);
	}
	
	public static void loadMS(String meanStdPath, Tensor mean, Tensor std) {
        try (RandomAccessFile File = new RandomAccessFile(meanStdPath, "r")) {
        	com.omega.engine.nn.network.utils.ModelUtils.loadParams(File, mean);
        	com.omega.engine.nn.network.utils.ModelUtils.loadParams(File, std);
            System.out.println("load mean std success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
	}
	
	public static void main(String[] args) {
		
		try {
			 
//			complate_ms();
			
			video_dit_train();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
