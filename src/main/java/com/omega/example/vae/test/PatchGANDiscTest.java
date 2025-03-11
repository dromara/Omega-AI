package com.omega.example.vae.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vqgan.PatchGANDiscriminator;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;

public class PatchGANDiscTest {
	
	
	public static void patchGANDisc() throws Exception {
		
		int batchSize = 16;
		int imageSize = 256;
		
		float[] mean = new float[] {0.5f, 0.5f,0.5f};
		float[] std = new float[] {0.5f, 0.5f,0.5f};
		
		int[] convChannels = new int[] {3, 64, 128, 256, 1};
		
		int[] kernels = new int[] {4, 4, 4, 4};
		
		int[] strides = new int[] {2, 2, 2, 1};
		
		int[] paddings = new int[] {1, 1, 1, 1};
		
		String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
		
		DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, true, mean, std);
		
		PatchGANDiscriminator network = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
		
		network.CUDNN = true;
		network.learnRate = 0.001f;
		
//		String lpipsWeight = "H:\\model\\lpips.json";
//		loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), network, true);
		
		MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

		optimizer.trainLPatchGANDisc(dataLoader);

	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			patchGANDisc();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
