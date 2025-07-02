package com.omega.example.vae.test;

import java.util.Map;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.TinyVQVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.nn.network.vqgan.PatchGANDiscriminator;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;

public class VQVAETest {
	
    public static void loadWeight(Map<String, Object> weightMap, TinyVQVAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * encoder block1
         *
         */
        ClipModelUtils.loadData(network.encoder.block1.conv.weight, weightMap, "encoder.0.0.weight");
        ClipModelUtils.loadData(network.encoder.block1.conv.bias, weightMap, "encoder.0.0.bias");
        network.encoder.block1.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.0.1.weight");
        network.encoder.block1.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.0.1.bias");
        network.encoder.block1.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.0.1.running_mean");
        network.encoder.block1.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.0.1.running_var");
        /**
         * encoder block2
         *
         */
        ClipModelUtils.loadData(network.encoder.block2.conv.weight, weightMap, "encoder.1.0.weight");
        ClipModelUtils.loadData(network.encoder.block2.conv.bias, weightMap, "encoder.1.0.bias");
        network.encoder.block2.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.1.1.weight");
        network.encoder.block2.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.1.1.bias");
        network.encoder.block2.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.1.1.running_mean");
        network.encoder.block2.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.1.1.running_var");
        /**
         * encoder block3
         *
         */
        ClipModelUtils.loadData(network.encoder.block3.conv.weight, weightMap, "encoder.2.0.weight");
        ClipModelUtils.loadData(network.encoder.block3.conv.bias, weightMap, "encoder.2.0.bias");
        network.encoder.block3.norm.gamma = ClipModelUtils.loadData(network.encoder.block1.norm.gamma, weightMap, 1, "encoder.2.1.weight");
        network.encoder.block3.norm.beta = ClipModelUtils.loadData(network.encoder.block1.norm.beta, weightMap, 1, "encoder.2.1.bias");
        network.encoder.block3.norm.runingMean = ClipModelUtils.loadData(network.encoder.block1.norm.runingMean, weightMap, 1, "encoder.2.1.running_mean");
        network.encoder.block3.norm.runingVar = ClipModelUtils.loadData(network.encoder.block1.norm.runingVar, weightMap, 1, "encoder.2.1.running_var");
        /**
         * pre_quant_conv
         *
         */
        network.pre_quant_conv.weight = ClipModelUtils.loadData(network.pre_quant_conv.weight, weightMap, 4, "preConv.weight");
        ClipModelUtils.loadData(network.pre_quant_conv.bias, weightMap, "preConv.bias");
        /**
         * embedding
         *
         */
        ClipModelUtils.loadData(network.embedding.weight, weightMap, "vq_layer.embedding.weight");
        /**
         * decoder input
         *
         */
        network.decoder.decoderInput.weight = ClipModelUtils.loadData(network.decoder.decoderInput.weight, weightMap, 4, "decoder_input.weight");
        ClipModelUtils.loadData(network.decoder.decoderInput.bias, weightMap, "decoder_input.bias");
        /**
         * decoder block1
         *
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
         *
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
         *
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

    public static void loadLPIPSWeight(Map<String, Object> weightMap, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
    }

    public static void vq_vae() {
        try {
            int batchSize = 16;
            int imageSize = 128;
            int latendDim = 4;
            int numLayers = 1;
            int headNum = 4;
            int num_vq_embeddings = 512;
            int[] downChannels = new int[]{32, 64, 128, 128};
            int[] midChannels = new int[]{128, 128};
            boolean[] downSample = new boolean[]{true, true, true};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, mean, std);
            VQVAE network = new VQVAE(LossType.MSE, UpdaterType.adamw, latendDim, imageSize, numLayers, headNum, num_vq_embeddings, downChannels, downSample, midChannels);
            network.CUDNN = true;
            network.learnRate = 0.0001f;
            //			String clipWeight = "H:\\model\\tiny_vae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainVQVAE(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vq_vae() {
        try {
            int batchSize = 24;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 8192;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, true, mean, std);
            TinyVQVAE network = new TinyVQVAE(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE(dataLoader);
            String save_model_path = "H:\\model\\vqvae500.model";
            ModelUtils.saveModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vq_vae_nature() {
        try {
            int batchSize = 32;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE network = new TinyVQVAE(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae() {
        try {
            int batchSize = 32;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 8;
            int num_vq_embeddings = 8192;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE network = new TinyVQVAE(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae2() {
        try {
            int batchSize = 4;
            int imageSize = 128;
            int z_dims = 64;
            int latendDim = 8;
            int num_vq_embeddings = 4096;
            int num_res_blocks = 1;
            int[] channels = new int[]{64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae_lpips() {
        try {
            int batchSize = 6;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE network = new TinyVQVAE(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE_lpips(dataLoader, lpips);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae2_lpips() {
        try {
            int batchSize = 4;
            int imageSize = 128;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 1;
            int[] channels = new int[]{64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            lpips.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2_lpips(dataLoader, lpips);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae2_lpips_gandisc() {
        try {
            int batchSize = 4;
            int imageSize = 128;
            int z_dims = 32;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 1;
            int[] channels = new int[]{64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            int[] convChannels = new int[]{3, 64, 128, 256, 1};
            int[] kernels = new int[]{4, 4, 4, 4};
            int[] strides = new int[]{2, 2, 2, 1};
            int[] paddings = new int[]{1, 1, 1, 1};
            PatchGANDiscriminator discriminator = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
            discriminator.CUDNN = true;
            discriminator.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            //			String save_model_path = "H:\\model\\vqvae2_128_500.model";
            //			ModelUtils.loadModel(network, save_model_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2_lpips_patchGANDisc(dataLoader, lpips, discriminator, 1500);
            String save_model_path = "H:\\model\\vqvae2_128_500.model";
            //			ModelUtils.saveModel(network, save_model_path);
            //
            ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae2_lpips_gandisc_32() {
        try {
            int batchSize = 4;
            int imageSize = 128;
            int z_dims = 32;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 2;
            int[] channels = new int[]{32, 64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            int[] convChannels = new int[]{3, 32, 64, 128, 256, 1};
            int[] kernels = new int[]{4, 4, 4, 4, 4};
            int[] strides = new int[]{2, 2, 2, 2, 1};
            int[] paddings = new int[]{1, 1, 1, 1, 1};
            PatchGANDiscriminator discriminator = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
            discriminator.CUDNN = true;
            discriminator.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            //			String save_model_path = "H:\\model\\vqvae2_128_500.model";
            //			ModelUtils.loadModel(network, save_model_path);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 2, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2_lpips_patchGANDisc(dataLoader, lpips, discriminator, 150000);
            //			String save_model_path = "H:\\model\\vqvae2_128_500.model";
            //			ModelUtils.saveModel(network, save_model_path);
            //
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void tiny_vqvae_lpips_gandisc() {
        try {
            int batchSize = 16;
            int imageSize = 128;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\LHQ256\\lhq_256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset128\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE network = new TinyVQVAE(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            int[] convChannels = new int[]{3, 64, 128, 256, 1};
            int[] kernels = new int[]{4, 4, 4, 4};
            int[] strides = new int[]{2, 2, 2, 1};
            int[] paddings = new int[]{1, 1, 1, 1};
            PatchGANDiscriminator discriminator = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
            discriminator.CUDNN = true;
            discriminator.learnRate = 0.001f;
            //			String clipWeight = "H:\\model\\tiny_vqvae.json";
            //			loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE_lpips_patchGANDisc(dataLoader, lpips, discriminator, 1500);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void anime_vqvae2_lpips_gandisc_32() {
        try {
            int batchSize = 2;
            int imageSize = 256;
            int z_dims = 32;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 1;
            int[] channels = new int[]{32, 64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            int[] convChannels = new int[]{3, 32, 64, 128, 256, 1};
            int[] kernels = new int[]{4, 4, 4, 4, 4};
            int[] strides = new int[]{2, 2, 2, 2, 1};
            int[] paddings = new int[]{1, 1, 1, 1, 1};
            PatchGANDiscriminator discriminator = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
            discriminator.CUDNN = true;
            discriminator.learnRate = 0.001f;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2_lpips_patchGANDisc(dataLoader, lpips, discriminator, 150000);
            String save_model_path = "/omega/models/anime_vqvae2_256.model";
            ModelUtils.saveModel(network, save_model_path);
            //
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void anime_vqvae2_lpips_gandisc_32_nogan() {
        try {
            int batchSize = 2;
            int imageSize = 256;
            int z_dims = 32;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 1;
            int[] channels = new int[]{32, 64, 128, 256};
            boolean[] attn_resolutions = new boolean[]{false, false, false, false};
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainTinyVQVAE2_lpips_nogan(dataLoader, lpips);
            String save_model_path = "/omega/models/anime_vqvae2_256.model";
            ModelUtils.saveModel(network, save_model_path);
            //
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void anime_vqvae2_lpips_gandisc_32_nogan2() {
        try {
            int batchSize = 16;
            int imageSize = 256;
            int z_dim = 128;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 2;
            int[] ch_mult = new int[]{1, 2, 2, 4};
            int ch = 32;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "D:\\dataset\\amine\\256\\";
//            String imgDirPath = "D:\\dataset\\pokemon\\dataset256\\";
//            String imgDirPath = "/omega/dataset/256/";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            VQVAE2 network = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dim, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
            network.CUDNN = true;
            network.gradCacheMode=true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "D:\\models\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainVQVAE2_lpips_nogan(dataLoader, lpips, "D:\\test\\vae\\256\\");
            String save_model_path = "/omega/models/anime_vqvae_256_v1.0.model";
            ModelUtils.saveModel(network, save_model_path);
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void pokemon_vqvae2_lpips_gandisc_32_nogan2() {
        try {
            int batchSize = 8;
            int imageSize = 256;
            int z_dim = 32;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 2;
            int[] ch_mult = new int[]{1, 2, 2, 4};
            int ch = 32;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
//            String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
            String imgDirPath = "D:\\dataset\\pokemon\\dataset256\\";
//            String imgDirPath = "/omega/dataset/256/";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            VQVAE2 network = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dim, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
            network.CUDNN = true;
            network.gradCacheMode=true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "D:\\models\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainVQVAE2_lpips_nogan(dataLoader, lpips, "D:\\test\\vae\\256\\");
            String save_model_path = "/omega/models/anime_vqvae_256_v1.0.model";
            ModelUtils.saveModel(network, save_model_path);
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void anime_vqvae2_lpips_hight() {
        try {
            int batchSize = 3;
            int imageSize = 256;
            int z_dims = 32;
            int latendDim = 16;
            int num_vq_embeddings = 512;
            int num_res_blocks = 1;
            int[] ch_mult = new int[]{1, 2, 2, 4};
            int ch = 32;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            VQVAE2 network = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
            network.CUDNN = true;
            network.gradCacheMode=true;
            network.learnRate = 0.001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            
            String model_path = "H:\\model\\pokemon_vqvae2_128.model";
            ModelUtils.loadModel(network, model_path);
            
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainVQVAE2_lpips_nogan(dataLoader, lpips, "D:\\test\\vae\\256\\");
            String save_model_path = "H:\\model\\pokemon_vqvae2_256.model";
            ModelUtils.saveModel(network, save_model_path);
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void anime_vqvae2_lpips_gandisc_32_gan2() {
        try {
            int batchSize = 8;
            int imageSize = 256;
            int z_dims = 64;
            int latendDim = 4;
            int num_vq_embeddings = 512;
            int num_res_blocks = 2;
            int[] ch_mult = new int[]{1, 2, 2, 4};
            int ch = 64;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "D:\\dataset\\amine\\256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
            VQVAE2 network = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
            network.CUDNN = true;
            network.gradCacheMode=true;
            network.learnRate = 0.0001f;
            LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "D:\\models\\lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
            lpips.CUDNN = true;
            int[] convChannels = new int[]{3, 64, 128, 256, 256, 1};
            int[] kernels = new int[]{4, 4, 4, 4, 4};
            int[] strides = new int[]{2, 2, 2, 2, 1};
            int[] paddings = new int[]{1, 1, 1, 1, 1};
            PatchGANDiscriminator discriminator = new PatchGANDiscriminator(LossType.MSE, UpdaterType.adamw, imageSize, convChannels, kernels, strides, paddings);
            discriminator.CUDNN = true;
            discriminator.learnRate = 0.0001f;
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainVQVAE2_lpips_patchGANDisc(dataLoader, lpips, discriminator, 200000);
            String save_model_path = "/omega/models/anime_vqvae2_256.model";
            ModelUtils.saveModel(network, save_model_path);
            //			ModelUtils.loadModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            //			vq_vae();
            //			tiny_vq_vae();
            //			tiny_vq_vae_nature();
            //			tiny_vqvae();
            //			tiny_vqvae2();
            //			tiny_vqvae2_lpips();
            //			tiny_vqvae_lpips();
            //			tiny_vqvae_lpips_gandisc();
            //			tiny_vqvae2_lpips_gandisc();
            //			tiny_vqvae2_lpips_gandisc_32();
            //			anime_vqvae2_lpips_gandisc_32();
            //			anime_vqvae2_lpips_gandisc_32_nogan();
//            anime_vqvae2_lpips_gandisc_32_nogan2();
//            anime_vqvae2_lpips_gandisc_32_nogan2();
//        	anime_vqvae2_lpips_gandisc_32_nogan2();
        	anime_vqvae2_lpips_gandisc_32_gan2();
//        	pokemon_vqvae2_lpips_gandisc_32_nogan2();
//            anime_vqvae2_lpips_hight();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
