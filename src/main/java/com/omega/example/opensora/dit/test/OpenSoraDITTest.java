package com.omega.example.opensora.dit.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.OpenSoraDIT;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BinDataType;

public class OpenSoraDITTest {
	
	public static void opensora_dit_train() throws Exception {
		
		String dataPath = "D:\\dataset\\wfvae\\video_latend.bin";
        String clipDataPath = "D:\\dataset\\wfvae\\video_clip.bin";

        int batchSize = 10;
        int latendDim = 8;
        int numFrames = 3;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * numFrames, height, width, 1, textEmbedDim, BinDataType.float32);
		
        int ditHeadNum = 12;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        int maxContextLen = 1;
        
    	OpenSoraDIT dit = new OpenSoraDIT(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, width, height, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true);
    	dit.CUDNN = true;
        dit.learnRate = 1e-4f;
         
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_opensora_dit_iddpm_kl(dataLoader, iddpm, "D:\\models\\opensora\\");
        String save_model_path = "D:\\models\\opensora\\opensora_dit.model";
        ModelUtils.saveModel(dit, save_model_path);
	}

	public static void main(String[] args) {
		
		try {
			 
			opensora_dit_train();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
