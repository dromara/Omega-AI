package com.omega.example.dit.test;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.OmegaDiT;
import com.omega.engine.nn.network.dit.OmegaDiT2;
import com.omega.engine.nn.network.vae.Flux_VAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.sd.utils.SDImageLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class OmegaDiT2Test {
	
	public static void omega_sprint_b1_iddpm_train_flux2vae() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_flux2vae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 128;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_224_224\\";
		boolean horizontalFilp = false;
        int imgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, imgDirPath, ".jpg", imgSize, imgSize, batchSize, horizontalFilp, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
		
		int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-5f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_16.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_Flux_Sprint_ICPlan_X(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt_flux//", 4);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_sprint_b1_iddpm_train_flux2vae_512() throws Exception {
		String dataPath = "/root/gpufree-data/2m/dalle_flux2vae_512_latend.bin";
        String clipDataPath = "/root/gpufree-data/2m/dalle_full_clip.bin";
		
        int batchSize = 10;
        int latendDim = 128;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        String labelPath = "/root/gpufree-data/processed_images/label.txt";
		String imgDirPath = "/root/gpufree-data/images_448/";
		boolean horizontalFilp = false;
        int imgSize = 448;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, imgDirPath, ".jpg", imgSize, imgSize, batchSize, horizontalFilp, mean, std);
		
        int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "/root/gpufree-data/omega/models/dionv2-14-b-512.model";
        ModelUtils.loadModel(dinov, repa_model_path);
		
		int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "/root/gpufree-data/omega/models/flux_sprint_b1_base.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_Flux_Sprint_ICPlan_X(dinov, dataLoader2, dataLoader, icplan, "/root/gpufree-data/models/omegaDiT2/", 1);
        String save_model_path = "/root/gpufree-data/models/omegaDiT2/fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_sprint_b1_iddpm_train_flux2vae_v() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_flux2vae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 128;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_224_224\\";
		boolean horizontalFilp = false;
        int imgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, imgDirPath, ".jpg", imgSize, imgSize, batchSize, horizontalFilp, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
		
		int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.00f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_20.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_Flux_Sprint_ICPlan_V(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt_flux//", 4);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }

	public static void omega_sprint_b1_iddpm_train_flux2vae_v_512() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_flux2vae_latend_512.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 5;
        int latendDim = 128;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_448_448\\";
		boolean horizontalFilp = false;
        int imgSize = 448;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, imgDirPath, ".jpg", imgSize, imgSize, batchSize, horizontalFilp, mean, std);
		
        int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b-512.model";
        ModelUtils.loadModel(dinov, repa_model_path);
		
		int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-6f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_1.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_Flux_Sprint_ICPlan_V(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt_flux//", 1);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void test_omega_sprint_path_drop_cfg_flux2vae() throws Exception {
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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int vaeLatendDim = 128;
        int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, vaeLatendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0);
        
//        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_16.model";
        String model_path = "D:\\models\\dit_txt_flux\\server_models\\flux_sprint_b1_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a highly detailed anime sexy beauty with big breasts.";
        labels[4] = "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.";
        labels[5] = "bright red phlox flowers bloom in a garden";
        labels[6] = "the cambridge shoulder bag";
        labels[7] = "A yellow mushroom grows in the forest";
        labels[8] = "a dog";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(label, 0, labels[0]);
        dataLoader.loadLabel_offset(label, 1, labels[1]);
        dataLoader.loadLabel_offset(label, 2, labels[2]);
        dataLoader.loadLabel_offset(label, 3, labels[3]);
        dataLoader.loadLabel_offset(label, 4, labels[4]);
        dataLoader.loadLabel_offset(label, 5, labels[5]);
        dataLoader.loadLabel_offset(label, 6, labels[6]);
        dataLoader.loadLabel_offset(label, 7, labels[7]);
        dataLoader.loadLabel_offset(label, 8, labels[8]);
        dataLoader.loadLabel_offset(label, 9, labels[9]);
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
            Tensor y_null = network.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
            }
        }
        
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A fox sleeping inside a large tansparent lightbule";
                labels[2] = "A beautiful girl with hair flowing like a cascading waterfail";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Game-Art - An island with different geographical properties and multiple small cities floating in space";
                labels[5] = "A man with short hair wearing a dark blazer and a patterned scarf is standing in front of a blurred background.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A cyberpunk panda is taking a walk on the street";
                labels[9] = "Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.";
                dataLoader.loadLabel_offset(label, 0, labels[0]);
                dataLoader.loadLabel_offset(label, 1, labels[1]);
                dataLoader.loadLabel_offset(label, 2, labels[2]);
                dataLoader.loadLabel_offset(label, 3, labels[3]);
                dataLoader.loadLabel_offset(label, 4, labels[4]);
                dataLoader.loadLabel_offset(label, 5, labels[5]);
                dataLoader.loadLabel_offset(label, 6, labels[6]);
                dataLoader.loadLabel_offset(label, 7, labels[7]);
                dataLoader.loadLabel_offset(label, 8, labels[8]);
                dataLoader.loadLabel_offset(label, 9, labels[9]);
                condInput = clip.get_full_clip_prompt_embeds(label);
        	}
        	
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_omega_sprint_path_drop_cfg_flux2vae_v() throws Exception {
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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int vaeLatendDim = 128;
        int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, vaeLatendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0);
        
//        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_0.model";
        String model_path = "D:\\models\\dit_txt_flux\\model_v\\flux_sprint_b1_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a highly detailed anime sexy beauty with big breasts.";
        labels[4] = "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.";
        labels[5] = "bright red phlox flowers bloom in a garden";
        labels[6] = "the cambridge shoulder bag";
        labels[7] = "A yellow mushroom grows in the forest";
        labels[8] = "a dog";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(label, 0, labels[0]);
        dataLoader.loadLabel_offset(label, 1, labels[1]);
        dataLoader.loadLabel_offset(label, 2, labels[2]);
        dataLoader.loadLabel_offset(label, 3, labels[3]);
        dataLoader.loadLabel_offset(label, 4, labels[4]);
        dataLoader.loadLabel_offset(label, 5, labels[5]);
        dataLoader.loadLabel_offset(label, 6, labels[6]);
        dataLoader.loadLabel_offset(label, 7, labels[7]);
        dataLoader.loadLabel_offset(label, 8, labels[8]);
        dataLoader.loadLabel_offset(label, 9, labels[9]);
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
            Tensor y_null = network.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
            }
        }
        
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A fox sleeping inside a large tansparent lightbule";
                labels[2] = "A beautiful girl with hair flowing like a cascading waterfail";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Game-Art - An island with different geographical properties and multiple small cities floating in space";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A cyberpunk panda is taking a walk on the street";
                labels[9] = "Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.";
                dataLoader.loadLabel_offset(label, 0, labels[0]);
                dataLoader.loadLabel_offset(label, 1, labels[1]);
                dataLoader.loadLabel_offset(label, 2, labels[2]);
                dataLoader.loadLabel_offset(label, 3, labels[3]);
                dataLoader.loadLabel_offset(label, 4, labels[4]);
                dataLoader.loadLabel_offset(label, 5, labels[5]);
                dataLoader.loadLabel_offset(label, 6, labels[6]);
                dataLoader.loadLabel_offset(label, 7, labels[7]);
                dataLoader.loadLabel_offset(label, 8, labels[8]);
                dataLoader.loadLabel_offset(label, 9, labels[9]);
                condInput = clip.get_full_clip_prompt_embeds(label);
        	}
        	
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_omega_sprint_path_drop_cfg_flux2vae_v_512() throws Exception {
		int imgSize = 512;
        int maxContextLen = 77;
        int batchSize = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);

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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int vaeLatendDim = 128;
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, vaeLatendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0);
        
        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_0.model";
//        String model_path = "D:\\models\\dit_txt_flux\\server_models\\flux_sprint_b1_2.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[28];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a girl wearning a white dress standing under the apple tree";
        labels[4] = "fruit cream cake";
        labels[5] = "bright red phlox flowers bloom in a garden";
        labels[6] = "the cambridge shoulder bag";
        labels[7] = "A yellow mushroom grows in the forest";
        labels[8] = "a dog";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        labels[10] = "A cat holding a sign that says hello world";
        labels[11] = "A man with short hair wearing a dark blazer and a patterned scarf is standing in front of a blurred background.";
        labels[12] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
        labels[13] = "porcelain girl's face. fine texture. surreal";
        labels[14] = "Half human, half robot, repaired human";
        labels[15] = "Poster of a mechanical cat, techical Schematics viewed from front.";
        labels[16] = "A beautiful girl with golden hair, cool and sunny";
        labels[17] = "future rocket station, intricate details, high resolution, unreal engine, UHD";
        labels[18] = "a car runing on the sand.";
        labels[19] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
        labels[20] = "a highly detailed anime sexy beauty with big breasts.";
        labels[21] = "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.";
        labels[22] = "A fox sleeping inside snow";
        labels[23] = "A beautiful girl with hair flowing like a cascading waterfail";
        labels[24] = "Game-Art - An island with different geographical properties and multiple small cities floating in space";
        labels[25] = "A cyberpunk panda is taking a walk on the street";
        labels[26] = "Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.";
        labels[27] = "One giant, sharp, metal square mirror in the center of the frame, four young people on the foreground, background sunny palm oil planation, tropical, realistic style, photography, nostalgic, green tone, mysterious, dreamy, bright color.";
        
        for(int i = 0;i<7;i++) {
			loadLabel_offset(bpe, label, 0, maxContextLen, labels[i * batchSize + 0]);
			loadLabel_offset(bpe, label, 1, maxContextLen, labels[i * batchSize + 1]);
			loadLabel_offset(bpe, label, 2, maxContextLen, labels[i * batchSize + 2]);
			loadLabel_offset(bpe, label, 3, maxContextLen, labels[i * batchSize + 3]);
            condInput = clip.get_full_clip_prompt_embeds(label);

            if(condInput_ynull == null) {
            	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
                Tensor y_null = network.main.labelEmbd.getY_embedding();
                int part_input_size = y_null.dataLength;
                for(int b = 0;b<batchSize;b++) {
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
                }
            }

        	for(int it = 0;it<5;it++) {
            	
            	System.out.println("start create test images.");

                GPUOP.getInstance().cudaRandn(noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

                Tensor result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_512\\"+i+"_" + it, result, mean, std);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);

                result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_512\\"+i+"_" + it + "_T", result, mean, std);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void test_omega_sprint_path_drop_cfg_flux2vae_512() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 512;
        int maxContextLen = 77;
        int batchSize = 4;
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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int vaeLatendDim = 128;
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, vaeLatendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0);
        
//        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_16.model";
        String model_path = "D:\\models\\dit_txt_flux\\model_v\\flux_sprint_b1_512_0_120000.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[28];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a girl wearning a white dress standing under the apple tree";
        labels[4] = "fruit cream cake";
        labels[5] = "bright red phlox flowers bloom in a garden";
        labels[6] = "the cambridge shoulder bag";
        labels[7] = "A yellow mushroom grows in the forest";
        labels[8] = "a dog";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        labels[10] = "A cat holding a sign that says hello world";
        labels[11] = "A man with short hair wearing a dark blazer and a patterned scarf is standing in front of a blurred background.";
        labels[12] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
        labels[13] = "porcelain girl's face. fine texture. surreal";
        labels[14] = "Half human, half robot, repaired human";
        labels[15] = "Poster of a mechanical cat, techical Schematics viewed from front.";
        labels[16] = "A beautiful girl with golden hair, cool and sunny";
        labels[17] = "future rocket station, intricate details, high resolution, unreal engine, UHD";
        labels[18] = "a car runing on the sand.";
        labels[19] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
        labels[20] = "a highly detailed anime sexy beauty with big breasts.";
        labels[21] = "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.";
        labels[22] = "A fox sleeping inside snow";
        labels[23] = "A beautiful girl with hair flowing like a cascading waterfail";
        labels[24] = "Game-Art - An island with different geographical properties and multiple small cities floating in space";
        labels[25] = "A cyberpunk panda is taking a walk on the street";
        labels[26] = "Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.";
        labels[27] = "One giant, sharp, metal square mirror in the center of the frame, four young people on the foreground, background sunny palm oil planation, tropical, realistic style, photography, nostalgic, green tone, mysterious, dreamy, bright color.";
        
        for(int i = 0;i<7;i++) {
			loadLabel_offset(bpe, label, 0, maxContextLen, labels[i * batchSize + 0]);
			loadLabel_offset(bpe, label, 1, maxContextLen, labels[i * batchSize + 1]);
			loadLabel_offset(bpe, label, 2, maxContextLen, labels[i * batchSize + 2]);
			loadLabel_offset(bpe, label, 3, maxContextLen, labels[i * batchSize + 3]);
            condInput = clip.get_full_clip_prompt_embeds(label);

            if(condInput_ynull == null) {
            	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
                Tensor y_null = network.main.labelEmbd.getY_embedding();
                int part_input_size = y_null.dataLength;
                for(int b = 0;b<batchSize;b++) {
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
                }
            }

        	for(int it = 0;it<5;it++) {
            	
            	System.out.println("start create test images.");

                GPUOP.getInstance().cudaRandn(noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

                Tensor result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_x_512\\"+i+"_" + it, result, mean, std);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);

                result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_x_512\\"+i+"_" + it + "_T", result, mean, std);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
	}
	
	public static void loadLabel_offset(BPETokenizerEN tokenizer, Tensor label, int index, int maxContextLen, String labelStr) {
    	int[] ids = tokenizer.encodeInt(labelStr, maxContextLen);
        for (int j = 0; j < maxContextLen; j++) {
            if (j < ids.length) {
                label.data[index * maxContextLen + j] = ids[j];
            } else {
                label.data[index * maxContextLen + j] = 0;
            }
        }
        label.hostToDevice();
    }
	
	public static void omega_b1_iddpm_train_flux2vae() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_flux2vae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 32;
        int latendDim = 128;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);

		int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT2 dit = new OmegaDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt_flux\\fluxvae_sprint_b1.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 40, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_Flux_Sprint_ICPlan_X(dataLoader, icplan, "D://models//dit_txt_flux//", 1);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void test_omega_cfg_flux2vae() throws Exception {
		
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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int vaeLatendDim = 128;
        int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT2 network = new OmegaDiT2(LossType.MSE, UpdaterType.adamw, vaeLatendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0);
        
        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_0.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a highly detailed anime sexy beauty with big breasts.";
        labels[4] = "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.";
        labels[5] = "bright red phlox flowers bloom in a garden";
        labels[6] = "the cambridge shoulder bag";
        labels[7] = "A yellow mushroom grows in the forest";
        labels[8] = "a dog";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(label, 0, labels[0]);
        dataLoader.loadLabel_offset(label, 1, labels[1]);
        dataLoader.loadLabel_offset(label, 2, labels[2]);
        dataLoader.loadLabel_offset(label, 3, labels[3]);
        dataLoader.loadLabel_offset(label, 4, labels[4]);
        dataLoader.loadLabel_offset(label, 5, labels[5]);
        dataLoader.loadLabel_offset(label, 6, labels[6]);
        dataLoader.loadLabel_offset(label, 7, labels[7]);
        dataLoader.loadLabel_offset(label, 8, labels[8]);
        dataLoader.loadLabel_offset(label, 9, labels[9]);
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
            Tensor y_null = network.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
            }
        }
        
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A fox sleeping inside a large tansparent lightbule";
                labels[2] = "A beautiful girl with hair flowing like a cascading waterfail";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Game-Art - An island with different geographical properties and multiple small cities floating in space";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A cyberpunk panda is taking a walk on the street";
                labels[9] = "Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.";
                dataLoader.loadLabel_offset(label, 0, labels[0]);
                dataLoader.loadLabel_offset(label, 1, labels[1]);
                dataLoader.loadLabel_offset(label, 2, labels[2]);
                dataLoader.loadLabel_offset(label, 3, labels[3]);
                dataLoader.loadLabel_offset(label, 4, labels[4]);
                dataLoader.loadLabel_offset(label, 5, labels[5]);
                dataLoader.loadLabel_offset(label, 6, labels[6]);
                dataLoader.loadLabel_offset(label, 7, labels[7]);
                dataLoader.loadLabel_offset(label, 8, labels[8]);
                dataLoader.loadLabel_offset(label, 9, labels[9]);
                condInput = clip.get_full_clip_prompt_embeds(label);
        	}
        	
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg_heun_step(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            OmegaDiTTest.showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_vae() {
		int imgSize = 256;
    	int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE2 vae = new Flux_VAE2(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\flux2_vae\\flux2_vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        for(int i = 0;i<4;i++) {

            Tensor latent = vae.encode(input);
            		
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
            MBSGDOptimizer.showImgs("D:\\test\\flux_vae\\_"+i, out, "test", mean, std);
        }
        
        
	}
	
    public static void main(String[] args) {
		 
        try {
           
//        	test_flux_vae();
        	
//        	omega_sprint_b1_iddpm_train_flux2vae();
        	
//        	omega_sprint_b1_iddpm_train_flux2vae_512();
        	
//        	test_omega_sprint_path_drop_cfg_flux2vae();
        	
//        	omega_b1_iddpm_train_flux2vae();
        	
//        	test_omega_cfg_flux2vae();
        	
//        	omega_sprint_b1_iddpm_train_flux2vae_v();
        	
//        	omega_sprint_b1_iddpm_train_flux2vae_v_512();
        	
//        	test_omega_sprint_path_drop_cfg_flux2vae_v();
        	
        	test_omega_sprint_path_drop_cfg_flux2vae_v_512();
        	
//        	test_omega_sprint_path_drop_cfg_flux2vae_512();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
  }
	
}
