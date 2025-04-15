package com.omega.test.vae.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.TinyVAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.utils.clip.ClipModelUtils;
import com.omega.utils.diffusion.DiffusionImageDataLoader;

import java.util.Map;

public class VAETest {
    public static void loadWeight(Map<String, Object> weightMap, TinyVAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * encoder block1

         */
        ClipModelUtils.loadData(network.encoder.block1.conv.weight, weightMap, "encoder.0.0.weight");
        ClipModelUtils.loadData(network.encoder.block1.conv.bias, weightMap, "encoder.0.0.bias");
        network.encoder.block1.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.0.1.weight");
        network.encoder.block1.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.0.1.bias");
        network.encoder.block1.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.0.1.running_mean");
        network.encoder.block1.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.0.1.running_var");
        /**
         * encoder block2

         */
        ClipModelUtils.loadData(network.encoder.block2.conv.weight, weightMap, "encoder.1.0.weight");
        ClipModelUtils.loadData(network.encoder.block2.conv.bias, weightMap, "encoder.1.0.bias");
        network.encoder.block2.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.1.1.weight");
        network.encoder.block2.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.1.1.bias");
        network.encoder.block2.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.1.1.running_mean");
        network.encoder.block2.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.1.1.running_var");
        /**
         * encoder block3

         */
        ClipModelUtils.loadData(network.encoder.block3.conv.weight, weightMap, "encoder.2.0.weight");
        ClipModelUtils.loadData(network.encoder.block3.conv.bias, weightMap, "encoder.2.0.bias");
        network.encoder.block3.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.2.1.weight");
        network.encoder.block3.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.2.1.bias");
        network.encoder.block3.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.2.1.running_mean");
        network.encoder.block3.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.2.1.running_var");
        /**
         * conv_mu

         */
        network.conv_mu.weight = ClipModelUtils.loadData(network.conv_mu.weight, weightMap, 4, "fc_mu.weight");
        ClipModelUtils.loadData(network.conv_mu.bias, weightMap, "fc_mu.bias");
        /**
         * conv_var

         */
        network.conv_var.weight = ClipModelUtils.loadData(network.conv_var.weight, weightMap, 4, "fc_var.weight");
        ClipModelUtils.loadData(network.conv_var.bias, weightMap, "fc_var.bias");
        /**
         * decoder input

         */
        network.decoder.decoderInput.weight = ClipModelUtils.loadData(network.decoder.decoderInput.weight, weightMap, 4, "decoder_input.weight");
        ClipModelUtils.loadData(network.decoder.decoderInput.bias, weightMap, "decoder_input.bias");
        /**
         * decoder block1

         */
        //		Tensor bw = new Tensor(256, 128, 3 ,3, true);
        //		ClipModelUtils.loadData(bw, weightMap, "decoder.0.0.weight");
        //		TensorOP.permute(bw, network.decoder.block1.conv.weight, new int[] {1, 0, 2, 3});
        ClipModelUtils.loadData(network.decoder.block1.conv.weight, weightMap, "decoder.0.0.weight");
        ClipModelUtils.loadData(network.decoder.block1.conv.bias, weightMap, "decoder.0.0.bias");
        network.decoder.block1.norm.gamma = ClipModelUtils.loadData(network.decoder.block1.norm.gamma, weightMap, 1, "decoder.0.1.weight");
        network.decoder.block1.norm.beta = ClipModelUtils.loadData(network.decoder.block1.norm.beta, weightMap, 1, "decoder.0.1.bias");
        network.decoder.block1.norm.runingMean = ClipModelUtils.loadData(network.decoder.block1.norm.runingMean, weightMap, 1, "decoder.0.1.running_mean");
        network.decoder.block1.norm.runingVar = ClipModelUtils.loadData(network.decoder.block1.norm.runingVar, weightMap, 1, "decoder.0.1.running_var");
        /**
         * decoder block2

         */
        //		Tensor bw2 = new Tensor(128, 64, 3 ,3, true);
        //		ClipModelUtils.loadData(bw2, weightMap, "decoder.1.0.weight");
        //		TensorOP.permute(bw2, network.decoder.block2.conv.weight, new int[] {1, 0, 2, 3});
        ClipModelUtils.loadData(network.decoder.block2.conv.weight, weightMap, "decoder.1.0.weight");
        ClipModelUtils.loadData(network.decoder.block2.conv.bias, weightMap, "decoder.1.0.bias");
        network.decoder.block2.norm.gamma = ClipModelUtils.loadData(network.decoder.block2.norm.gamma, weightMap, 1, "decoder.1.1.weight");
        network.decoder.block2.norm.beta = ClipModelUtils.loadData(network.decoder.block2.norm.beta, weightMap, 1, "decoder.1.1.bias");
        network.decoder.block2.norm.runingMean = ClipModelUtils.loadData(network.decoder.block2.norm.runingMean, weightMap, 1, "decoder.1.1.running_mean");
        network.decoder.block2.norm.runingVar = ClipModelUtils.loadData(network.decoder.block2.norm.runingVar, weightMap, 1, "decoder.1.1.running_var");
        /**
         * decoder block3

         */
        //		Tensor bw3 = new Tensor(64, 3, 3 ,3, true);
        //		ClipModelUtils.loadData(bw3, weightMap, "decoder.2.0.weight");
        //		TensorOP.permute(bw3, network.decoder.block3.conv.weight, new int[] {1, 0, 2, 3});
        ClipModelUtils.loadData(network.decoder.block3.conv.weight, weightMap, "decoder.2.0.weight");
        ClipModelUtils.loadData(network.decoder.block3.conv.bias, weightMap, "decoder.2.0.bias");
        network.decoder.block3.norm.gamma = ClipModelUtils.loadData(network.decoder.block3.norm.gamma, weightMap, 1, "decoder.2.1.weight");
        network.decoder.block3.norm.beta = ClipModelUtils.loadData(network.decoder.block3.norm.beta, weightMap, 1, "decoder.2.1.bias");
        network.decoder.block3.norm.runingMean = ClipModelUtils.loadData(network.decoder.block3.norm.runingMean, weightMap, 1, "decoder.2.1.running_mean");
        network.decoder.block3.norm.runingVar = ClipModelUtils.loadData(network.decoder.block3.norm.runingVar, weightMap, 1, "decoder.2.1.running_var");
    }

    public static void tiny_vae() {
        try {
            int batchSize = 8;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 4;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, mean, std);
            TinyVAE network = new TinyVAE(LossType.MSE_SUM, UpdaterType.adamw, z_dims, latendDim, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVAE(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            tiny_vae();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

