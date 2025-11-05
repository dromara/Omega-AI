package com.omega.example.dit.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.DiT_ORG;
import com.omega.engine.nn.network.dit.PixArtDiT;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

public class PixArtTest {
	
	public static void train() throws Exception {
		
		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        PixArtDiT dit = new PixArtDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

        optimizer.train_pixart_iddpm(dataLoader, iddpm, "D:\\test\\models\\mmdit\\", 0.13025f);
        String save_model_path = "/omega/models/pixart_dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
		
	}
	
	public static void train_rope() throws Exception {
		
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 30;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        float y_prob = 0.0f;
        
        PixArtDiT dit = new PixArtDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

        optimizer.train_pixart_iddpm(dataLoader, iddpm, "D:\\test\\models\\mmdit\\", 0.13025f);
        String save_model_path = "/omega/models/pixart_dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
		
	}
	
	public static void test() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        PixArtDiT network = new PixArtDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, 0.0f);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);
        
        String model_path = "D:\\test\\models\\mmdit\\pixart_dit_4.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);

        Tensor mean_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor var_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
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
            MBSGDOptimizer.testDiT_IDDPM(i+"_pixart", latend, noise, t, condInput, mean_l, var_l, network, vae, iddpm, labels, "D:\\test\\dit4\\pixart\\", 0.13025f);
            System.out.println("finish create.");
        }

    }
	
	public static void test_rope() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        PixArtDiT network = new PixArtDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, 0.0f);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);
        
        String model_path = "D:\\test\\models\\mmdit\\pixart_dit_8.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);

        Tensor mean_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor var_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[10];
        for(int i = 0;i<4;i++) {
        	System.out.println("start create test images.");
            labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
            labels[3] = "a little girl standing on the beach";
            labels[4] = "fruit cream cake";
            labels[5] = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k";
            
            labels[6] = "A small, fluffy white and black dog is wrapped in a white towel, lying on a bed with a floral pattern.";
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
            MBSGDOptimizer.testDiT_IDDPM(i+"_pixart", latend, noise, t, condInput, cos, sin, mean_l, var_l, network, vae, iddpm, labels, "D:\\test\\dit4\\pixart\\", 0.13025f);
            System.out.println("finish create.");
        }

    }
	
	public static void test_rope2() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        float y_prob = 0;
        
        DiT_ORG network = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        network.RUN_MODEL = RunModel.TEST;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, network.cudaManager);
        
        String model_path = "D:\\test\\models\\dit_xl2\\dit_xl2_18.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);

        Tensor mean_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor var_l = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[10];
        for(int i = 0;i<4;i++) {
        	System.out.println("start create test images.");
            labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
            labels[3] = "a little girl standing on the beach";
            labels[4] = "fruit cream cake";
            labels[5] = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k";
            
            labels[6] = "A dog fly on the sky.";
            
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
            MBSGDOptimizer.testDiT_IDDPM(i+"_pixart", latend, noise, t, condInput, cos, sin, mean_l, var_l, network, vae, iddpm, labels, "D:\\test\\dit4\\dit_org\\", 0.13025f);
            System.out.println("finish create.");
        }

    }
	
	public static void main(String[] args) {
		 
        try {
        	
//        	train();
        	
//        	train_rope();
        	
//        	test();
        	
//        	test_rope();
        	
        	test_rope2();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
	}
}
