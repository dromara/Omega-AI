package com.omega.example.vae.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.DC_AE;
import com.omega.engine.nn.network.vae.TinyVQVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.nn.network.vqgan.PatchGANDiscriminator;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;

import java.util.Map;

public class DC_AETest {

    public static void dcae_lpips() {
        try {
            int batchSize = 2;
            int imageSize = 128;
            int initChannel = 16;
            int latent_channels = 32;
            int num_blocks = 1;
            int spatial_compression = 32;
            
            int group = 8;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            
            DC_AE network = new DC_AE(LossType.MSE, UpdaterType.adamw, initChannel, latent_channels, imageSize, num_blocks, spatial_compression, group);

            network.CUDNN = true;
            network.gradCacheMode=true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainDCAE_lpips(dataLoader, lpips);
//            String save_model_path = "/omega/models/anime_vqvae2_256.model";
//            ModelUtils.saveModel(network, save_model_path);
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            
            dcae_lpips();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
