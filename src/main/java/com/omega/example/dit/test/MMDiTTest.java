package com.omega.example.dit.test;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.MMDiT;
import com.omega.engine.nn.network.MMDiT_RoPE;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

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
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false);
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
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false);
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
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, y_prob);
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
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm2(dataLoader, vae, clip, iddpm, "D://test//dit4//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_iddpm_amine_train_by_latend() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 24;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        MMDiT dit = new MMDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_iddpm(dataLoader, iddpm, "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void mmdit_rope_iddpm_amine_train_by_latend() throws Exception {
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 32;
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
        
        MMDiT_RoPE dit = new MMDiT_RoPE(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_MMDiT_RoPE_iddpm(dataLoader, iddpm, "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
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
	
	
	
	public static void main(String[] args) {
		 
        try {

//        	mmdit_iddpm_amine_train();
        	
//        	mmdit_iddpm_amine_train2();
        	
//        	mmdit_rope_iddpm_amine_train();
        	
//        	mmdit_rope_iddpm_amine_train2();
        	
//        	testClip();
        	
//        	mmdit_rope_iddpm_amine_train_by_latend();
        	
        	mmdit_iddpm_amine_train_by_latend();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
   }
	
}
