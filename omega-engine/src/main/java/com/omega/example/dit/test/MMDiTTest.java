package com.omega.example.dit.test;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.MMDiT;
import com.omega.engine.nn.network.dit.MMDiT_RoPE;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.driver.JCudaDriver;

public class MMDiTTest {
	
	public static void mmdit_iddpm_amine_train() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int time = maxContextLen;
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 8;
        int n_layers = 12;
        int textEmbedDim = 512;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, n_layers);
        clip.CUDNN = true;
        clip.time = time;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\clip-vit-base-patch32.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, false);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_iddpm(dataLoader, vae, clip, iddpm, "D://test//dit2//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_iddpm_amine_train2() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_iddpm2(dataLoader, vae, clip, iddpm, "D://test//dit2//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int time = maxContextLen;
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 8;
        int n_layers = 12;
        int textEmbedDim = 512;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, n_layers);
        clip.CUDNN = true;
        clip.time = time;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\clip-vit-base-patch32.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm(dataLoader, vae, clip, iddpm, "D://test//dit2//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train2() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\CLIP-GmP-ViT-L-14\\vocab.json";
        String mergesPath = "D:\\models\\CLIP-GmP-ViT-L-14\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
       
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm2(dataLoader, vae, clip, iddpm, "D://test//dit4//", "/omega/models/dit/", 0.13025f, 11, 0.25f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_iddpm_amine_train_by_latend() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 100, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_iddpm(dataLoader, iddpm, "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_iddpm_amine_train_by_latend_dispLoss() throws Exception {
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_iddpm_disp_loss(dataLoader, iddpm, "/omega/models/dit/", 0.13025f, 11, 0.25f);
        String save_model_path = "/omega/models/mmdit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train_by_latend() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.3f;
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm(dataLoader, iddpm, "D:\\test\\models\\mmdit\\", 1f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train_by_latend_dispLoss() throws Exception {
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm_disp_loss(dataLoader, iddpm, "/omega/models/dit/", 0.13025f, 11, 0.25f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train_by_latend_kl_dispLoss() throws Exception {
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 4;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_rope_iddpm_kl_disp_loss(dataLoader, iddpm, "/omega/models/dit/", 0.13025f, 11, 0.25f);
        String save_model_path = "/omega/models/mmdit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }

	public static void testClip() {
		
		int batchSize = 2;
		
		String vocabPath = "D:\\models\\CLIP-GmP-ViT-L-14\\vocab.json";
        String mergesPath = "D:\\models\\CLIP-GmP-ViT-L-14\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
		
	 	int maxContextLen = 77;
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        String[] texts = new String[] {
                "A photo of a Maine Coon cat with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. It is wearing a miniature top hat. the cat is holding a sign made from weathered wood. Written on the sign in hand-painted script are the words 'long CLIP is long and long CAT is lovely!'.",
        		"A photo of a Maine Coon cat with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. It is wearing a miniature top hat. the cat is holding a sign made from weathered wood. Written on the sign in hand-painted script are the words 'long CLIP is long and long CAT is lovely!'. Background grass with wildflowers."
        };
        
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);

        for(int i = 0;i<batchSize;i++) {
        	int[] ids = bpe.encodeInt(texts[i], maxContextLen);
            System.err.println(JsonUtils.toJson(ids));
            float eos_id = 0;
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = bpe.eos();
                }
                //获取第一个结束符位置
                if(label.data[i * maxContextLen + j] == bpe.eos() && eos_id == 0) {
                	eos_id = j;
                }
            }
            eosIds.data[i] = eos_id;
        }

        label.hostToDevice();
        eosIds.hostToDevice();
        
        eosIds.showDM("eosIds");
        
        Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
        
        clip.get_clip_prompt_embeds(label, eosIds, condInput);
        
        condInput.showDM();
        
        PrintUtils.printImage(condInput);
	}
	
	public static void testLoadData() {
		
		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 80;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
		
        Tensor latend = new Tensor(batchSize, dataLoader.channel, dataLoader.height, dataLoader.width, true);
        Tensor condInput = new Tensor(batchSize , 1, 1, dataLoader.clipEmbd, true);
        
        for (int i = 0; i < 1; i++) {

            int[][] indexs = dataLoader.shuffle();

            dataLoader.loadData(indexs[0], latend, condInput, 0);
            
            for (int it = 0; it < dataLoader.count_it; it++) {
            	dataLoader.loadData(indexs[it], latend, condInput, it);
            }
            
        }
        
	}
	
	public static void testVAVAE() {
		int imgSize = 256;
		
        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int batchSize = 4;
        int imageSize = 256;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, true, mean, std);
        
        int[] indexs = new int[]{3960, 145, 2, 9876};
//        int[] indexs = new int[] {145};
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        Tensor latent = vae.encode(input);
        //		latent.showDM("latent");
        latent.showShape();
        //		Tensor latent = new Tensor(batchSize, 4, 32, 32, RandomUtils.gaussianRandom(batchSize * 4 * 32 * 32, 1.0f, 0.0f), true);
        Tensor out = vae.decode(latent);
        //		out.showDM("out");
        out.showShape();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         */
        MBSGDOptimizer.showImgs("D:\\test\\va_vae\\", out, "test", mean, std);
        
	}
	
	public static void mmdit_iddpm_amine_test() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 6;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, true);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        String model_path = "D:\\test\\models\\mmdit\\anime_dit_20.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, dit.textEmbedDim, true);
        
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        Tensor score = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        Tensor latend = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);

        dit.RUN_MODEL = RunModel.TEST;
        System.out.println("start create test images.");
        String[] labels = new String[6];
        labels[0] = "A young girl in cap and shorts with letter 'W' swing in the bule water, surrounded by a dynamic energy.";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "A woman fly in the sky, wearing a red top with a sheer overlay and blue jeans.";
        labels[3] = "a 3d close-up of stylized white flowers with intricate details, lush leaves, and foliage, bathed in dramatic chiaroscuro lighting against a dark background, high resolution, sharp focus.";
        labels[4] = "a majestic tiger face in a lush, natural landscape, with intricate fur details and rich textures, bathed in soft golden hour light, high resolution, photorealistic.";
        labels[5] = "A lovely corgi was walking on the bottom of the sea. There was a turtle swimming beside him. There were many colorful fish around";
        dataLoader.loadLabel_offset(label, 0, labels[0], eosIds);
        dataLoader.loadLabel_offset(label, 1, labels[1], eosIds);
        dataLoader.loadLabel_offset(label, 2, labels[2], eosIds);
        dataLoader.loadLabel_offset(label, 3, labels[3], eosIds);
        dataLoader.loadLabel_offset(label, 4, labels[4], eosIds);
        dataLoader.loadLabel_offset(label, 5, labels[5], eosIds);
        condInput = clip.get_clip_prompt_embeds(label, eosIds, condInput);
        MBSGDOptimizer.testDiT_IDDPM(10 + "", latend, noise, t, condInput, score, dit, vae, iddpm, labels, "D:\\test\\dit4\\", 0.13025f);
        System.out.println("finish create.");
    }
	
	public static void mmdit_iddpm_amine_test2() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 6;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, true);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        String model_path = "D:\\test\\models\\mmdit\\anime_dit_20.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, dit.textEmbedDim, true);
        
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        Tensor score = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        Tensor latend = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);

        dit.RUN_MODEL = RunModel.TEST;
        
        for(int i = 0;i<10;i++) {
            System.out.println("start create test images.");
            String[] labels = new String[6];
            labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
            labels[3] = "a little girl standing on the beach";
            labels[4] = "fruit cream cake";
            labels[5] = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k";
            dataLoader.loadLabel_offset(label, 0, labels[0], eosIds);
            dataLoader.loadLabel_offset(label, 1, labels[1], eosIds);
            dataLoader.loadLabel_offset(label, 2, labels[2], eosIds);
            dataLoader.loadLabel_offset(label, 3, labels[3], eosIds);
            dataLoader.loadLabel_offset(label, 4, labels[4], eosIds);
            dataLoader.loadLabel_offset(label, 5, labels[5], eosIds);
            condInput = clip.get_clip_prompt_embeds(label, eosIds, condInput);
            MBSGDOptimizer.testDiT_IDDPM((20+i) + "2", latend, noise, t, condInput, score, dit, vae, iddpm, labels, "D:\\test\\dit4\\test2\\", 0.13025f);
            System.out.println("finish create.");
        }
        
    }
	
	public static void mmdit_rope_iddpm_amine_test() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 6;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE network = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, true, y_prob);
        network.CUDNN = true;
        network.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);

        String model_path = "D:\\test\\models\\mmdit\\anime_dit_20.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, network.textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor mean_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor var_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[6];
        System.out.println("start create test images.");
        labels[0] = "A young girl in cap and shorts with letter 'W' swing in the bule water, surrounded by a dynamic energy.";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "A woman fly in the sky, wearing a red top with a sheer overlay and blue jeans.";
        labels[3] = "a 3d close-up of stylized white flowers with intricate details, lush leaves, and foliage, bathed in dramatic chiaroscuro lighting against a dark background, high resolution, sharp focus.";
        labels[4] = "a majestic tiger face in a lush, natural landscape, with intricate fur details and rich textures, bathed in soft golden hour light, high resolution, photorealistic.";
        labels[5] = "A lovely corgi was walking on the bottom of the sea. There was a turtle swimming beside him. There were many colorful fish around";
        dataLoader.loadLabel_offset(label, 0, labels[0], eosIds);
        dataLoader.loadLabel_offset(label, 1, labels[1], eosIds);
        dataLoader.loadLabel_offset(label, 2, labels[2], eosIds);
        dataLoader.loadLabel_offset(label, 3, labels[3], eosIds);
        dataLoader.loadLabel_offset(label, 4, labels[4], eosIds);
        dataLoader.loadLabel_offset(label, 5, labels[5], eosIds);
        condInput = clip.get_clip_prompt_embeds(label, eosIds, condInput);
        MBSGDOptimizer.testDiT_IDDPM(20 + "", latend, noise, t, condInput, cos, sin, mean_l, var_l, network, vae, iddpm, labels, "D:\\test\\dit4\\", 0.13025f);
        System.out.println("finish create.");
    }
	
	public static void mmdit_rope_iddpm_amine_test2() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 6;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE network = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);

        String model_path = "D:\\test\\models\\mmdit\\anime_mmdit_400.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, network.textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor score = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[6];
        for(int i = 0;i<10;i++) {
	        System.out.println("start create test images.");
	        labels[0] = "A cat holding a sign that says hello world";
	        labels[1] = "a vibrant anime mountain lands";
	        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
	        labels[3] = "a little girl standing on the beach";
	        labels[4] = "fruit cream cake";
	        labels[5] = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k";
	        dataLoader.loadLabel_offset(label, 0, labels[0], eosIds);
	        dataLoader.loadLabel_offset(label, 1, labels[1], eosIds);
	        dataLoader.loadLabel_offset(label, 2, labels[2], eosIds);
	        dataLoader.loadLabel_offset(label, 3, labels[3], eosIds);
	        dataLoader.loadLabel_offset(label, 4, labels[4], eosIds);
	        dataLoader.loadLabel_offset(label, 5, labels[5], eosIds);
	        condInput = clip.get_clip_prompt_embeds(label, eosIds, condInput);
	        MBSGDOptimizer.testDiT_IDDPM(2 + "_"+i, latend, noise, t, condInput, cos, sin, score, network, vae, iddpm, labels, "D:\\test\\dit4\\test2\\", 0.13025f);
	        System.out.println("finish create.");
        }
    }
	
	public static void mmdit_rope_iddpm_amine_test3() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        MMDiT_RoPE network = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);

        String model_path = "D:\\test\\models\\mmdit\\anime_mmdit_3.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, network.textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor score = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latend_mean = new Tensor(4, 1, 1, 1, new float[] {1.2692487f,0.53773075f,-0.30631456f,1.6842138f}, true);
        Tensor latend_std = new Tensor(4, 1, 1, 1, new float[] {9.447528f,6.495828f,7.30328f,6.17809f}, true);
        
        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[10];
        for(int i = 0;i<10;i++) {
	        System.out.println("start create test images.");
	        labels[0] = "A cat holding a sign that says hello world";
	        labels[1] = "a vibrant anime mountain lands";
	        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
	        labels[3] = "a little girl standing on the beach";
	        labels[4] = "fruit cream cake";
	        labels[5] = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k";
	        
	        labels[6] = "A panda sleep on the water.";
            labels[7] = "A woman with shoulder-length blonde hair wearing a dark blouse with a floral patterned collar.";
            labels[8] = "A small, grey crochet plush toy of a cat with pink paws and a pink nose sits on a wooden surface.";
            labels[9] = "A group of humpback whales is swimming in the ocean, with one whale prominently in the foreground and two others in the background. The water is clear, and the whales are surrounded by a multitude of bubbles, creating a dynamic underwater scene.";
	        dataLoader.loadLabel_offset(label, 0, labels[0], eosIds);
	        dataLoader.loadLabel_offset(label, 1, labels[1], eosIds);
	        dataLoader.loadLabel_offset(label, 2, labels[2], eosIds);
	        dataLoader.loadLabel_offset(label, 3, labels[3], eosIds);
	        dataLoader.loadLabel_offset(label, 4, labels[4], eosIds);
	        dataLoader.loadLabel_offset(label, 5, labels[5], eosIds);
	        dataLoader.loadLabel_offset(label, 6, labels[6], eosIds);
            dataLoader.loadLabel_offset(label, 7, labels[7], eosIds);
            dataLoader.loadLabel_offset(label, 8, labels[8], eosIds);
            dataLoader.loadLabel_offset(label, 9, labels[9], eosIds);
	        condInput = clip.get_clip_prompt_embeds(label, eosIds, condInput);
	        MBSGDOptimizer.testDiT_IDDPM(3 + "_"+i, latend, noise, t, condInput, cos, sin, score, network, vae, iddpm, labels, "D:\\test\\dit4\\test2\\", latend_mean, latend_std, 0.13025f);
	        System.out.println("finish create.");
        }
    }
	
	public static void main(String[] args) {
		 
        try {

//        	mmdit_iddpm_amine_train();
        	
//        	mmdit_iddpm_amine_train2();
        	
//        	mmdit_rope_iddpm_amine_train();
        	
//        	mmdit_rope_iddpm_amine_train2();
        	
//        	testClip();
        	
//        	mmdit_rope_iddpm_amine_train_by_latend();
        	
//        	mmdit_iddpm_amine_train_by_latend();
        	
//        	testLoadData();
        	
//        	mmdit_iddpm_amine_test();
        	
//        	mmdit_iddpm_amine_test2();
        	
//        	mmdit_iddpm_amine_train_by_latend_dispLoss();
        	
//        	mmdit_rope_iddpm_amine_train_by_latend_dispLoss();
        	
//        	mmdit_rope_iddpm_amine_train_by_latend_kl_dispLoss();
        	
//        	mmdit_rope_iddpm_amine_test();
        	
//        	mmdit_rope_iddpm_amine_test2();
        	
        	testVAVAE();
        	
//        	mmdit_rope_iddpm_amine_train_by_latend();
        	
//        	mmdit_rope_iddpm_amine_test3();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
   }
	
}
