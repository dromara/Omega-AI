package com.omega.example.dit.test;

import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dinovision.NestedTensorBlock;
import com.omega.engine.nn.layer.dit.flux.FluxDiTBlock2;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.FluxDiT;
import com.omega.engine.nn.network.dit.FluxDiT2;
import com.omega.engine.nn.network.dit.FluxDiT3;
import com.omega.engine.nn.network.dit.FluxDiT_REG;
import com.omega.engine.nn.network.dit.FluxDiT_REPA;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT;
import com.omega.engine.nn.network.dit.OmegaDiT;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT3;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT4;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT_REG;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.dit.utils.RandomMaskUtils;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.runtime.JCuda;

public class FluxDiTTest {

	public static void flux_dit_b2_iddpm_train() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
//		String clipDataPath = "D:\\dataset\\amine\\vavae_2clip.bin";
		
        int batchSize = 32;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 16;
        int en_depth = 2;
        int de_depth = 10;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        FluxDiT2 dit = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.00004f;
        
        FluxDiT2 dit_ema = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_30.model";
        ModelUtils.loadModel(dit, model_path);
        String model_ema_path = "D:\\models\\dit_txt\\flux_ddt_b1_ema30.model";
        ModelUtils.loadModel(dit_ema, model_ema_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 50, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux2_ICPlan_resume(dit_ema, dataLoader, icplan, "D://models//dit_txt//", mean, std, 1f, 10);
//        optimizer.train_Flux2_ICPlan(dit_ema, dataLoader, icplan, "D://models//dit_txt//", mean, std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_repa_b1_iddpm_train() throws Exception {
		
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        
        int z_idx = 4;
        
        float y_prob = 0.1f;
        
        FluxDiT_REPA dit = new FluxDiT_REPA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, z_idx, dinov_hiddenSize, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_4.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_REPA_ICPlan(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_reg_b1_iddpm_train() throws Exception {
		
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        
        int z_idx = 4;
        int cls_dim = 768;
        
        float y_prob = 0.1f;
        
        FluxDiT_REG dit = new FluxDiT_REG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, z_idx, dinov_hiddenSize, cls_dim, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_20.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_REG_ICPlan(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_sprint_b1_iddpm_train() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        
        FluxDiT_SPRINT dit = new FluxDiT_SPRINT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 8e-5f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt\\flux_sprint_b1_36.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }

	public static void flux_sprint_b1_iddpm_train2() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        dit.learnRate = 2e-5f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt2\\flux_sprint_b1_28.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan2(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_sprint_b1_iddpm_train2_512() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend_512.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 5;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        dit.learnRate = 5e-5f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt2_512\\flux_sprint_b1_512_2_5000.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan2_512(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2_512//", latend_mean, latend_std, 1f, 1);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1_512.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_sprint_b1_iddpm_train3() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 36;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        float token_drop = 0.75f;
        float path_drop_prob = 0.05f;
        
        FluxDiT_SPRINT3 dit = new FluxDiT_SPRINT3(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-6f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt3\\flux_sprint_b1_32.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan3(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt3//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_sprint_b1_iddpm_train4() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        
        FluxDiT_SPRINT4 dit = new FluxDiT_SPRINT4(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt2\\flux_sprint_b1_20.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan4(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_sprint_reg_b1_iddpm_train() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 32;
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
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContext, batchSize, horizontalFilp, mean, std);
		
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
        
        FluxDiT_SPRINT_REG dit = new FluxDiT_SPRINT_REG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt\\flux_sprint_b1_36.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_Reg_ICPlan(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_dit_l1_iddpm_train() throws Exception {
		String dataPath = "/root/gpufree-data/txt2img_2m/vavae_2m_latend.bin";
        String clipDataPath = "/root/gpufree-data/txt2img_2m/vavae_clip.bin";
//		String clipDataPath = "D:\\dataset\\amine\\vavae_2clip.bin";
		
        int batchSize = 24;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 16;
        int latendSize = 16;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1024;
        
        float y_prob = 0.1f;
        
        FluxDiT dit = new FluxDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0002f;
        
        FluxDiT dit_ema = new FluxDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//      String model_path = "D:\\models\\dit_txt\\flux_dit_xl1_24.model";
//      ModelUtils.loadModel(dit, model_path);
//	    String model_ema_path = "D:\\models\\dit_txt\\flux_dit_xl1_24.model";
//	    ModelUtils.loadModel(dit_ema, model_ema_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
//        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.5908793f,0.08059605f,-0.4490295f,-0.36698666f,-0.09954782f,-1.5645596f,-0.4480346f,-0.27834356f,0.16410087f,0.6236304f,-0.722689f,1.9468695f,0.03337372f,0.67487925f,0.43168893f,1.7030053f,-1.1693488f,1.803961f,-0.26420984f,-0.64909077f,0.5674515f,-0.020895006f,-1.6284368f,0.62391245f,2.782418f,2.1002185f,-0.47597224f,0.056646377f,-1.3163285f,-0.37474704f,0.61040056f,-0.2833984f}, true);
//        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.094117f,4.0699778f,3.4177587f,3.6069686f,3.599577f,3.2107027f,3.0169837f,3.484884f,3.7302747f,3.6297033f,3.6147742f,3.5576963f,3.7799015f,3.6828954f,3.2159526f,3.5250695f,3.8837016f,3.5258102f,3.6749682f,3.2791677f,3.394157f,3.001622f,3.5698154f,4.284372f,2.5654256f,3.3422892f,4.98195f,3.1721509f,3.2045052f,3.8944986f,4.281285f,4.121331f}, true);

        optimizer.train_Flux_ICPlan2(dit_ema, dataLoader, icplan, "/root/gpufree-data/models/", mean, std, 1f, 20000);
//        optimizer.train_Flux_ICPlan2_resume(dit_ema, dataLoader, icplan, "/root/gpufree-data/models/", mean, std, 1f, 20000);
        String save_model_path = "/omega/models/flux_l1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void flux_dit_b2_iddpm_train_unsample() throws Exception {
		String dataPath = "D:\\dataset\\amine\\vavae_latend_m_s.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
//		String clipDataPath = "D:\\dataset\\amine\\vavae_2clip.bin";
		
        int batchSize = 32;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * 2, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 16;
        int en_depth = 2;
        int de_depth = 10;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        FluxDiT2 dit = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_36.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux2_ICPlan_unsample(dataLoader, icplan, "D://models//dit_txt//", mean, std, 1f, 4);
        String save_model_path = "/omega/models/dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void test_flux_cfg() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 8;
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 16;
        int latendSize = 16;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1024;
        
        float y_prob = 0.1f;
        
        FluxDiT network = new FluxDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\dit_vavae\\flux\\flux_dit_l1_ema3_60000.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.57625645f,0.09338784f,-0.42964765f,-0.37670407f,-0.08795908f,-1.5591346f,-0.44817135f,-0.30657783f,0.13237366f,0.65041465f,-0.7171756f,1.9741942f,0.028869499f,0.6699318f,0.42782572f,1.7180899f,-1.1657567f,1.8240571f,-0.27507848f,-0.64780587f,0.57183015f,-0.015657624f,-1.6581186f,0.64075494f,2.8023534f,2.1147559f,-0.46823075f,0.06350619f,-1.3322849f,-0.3576193f,0.589975f,-0.28093442f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.092015f,4.054383f,3.416052f,3.608325f,3.6037152f,3.1869633f,3.0185716f,3.4761536f,3.730099f,3.6259832f,3.608243f,3.562f,3.7766607f,3.6814737f,3.208086f,3.5293725f,3.8835256f,3.5182753f,3.679344f,3.277554f,3.3914754f,3.0021727f,3.5673943f,4.2805233f,2.5594256f,3.3488555f,4.982151f,3.1698997f,3.1933618f,3.8946505f,4.275168f,4.1173005f}, true);

//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat holding a sign that says hello world";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
        labels[4] = "Half human, half robot, repaired human";
        labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
        labels[6] = "A beautiful girl with golden hair, cool and sunny";
        labels[7] = "A dog";
//        labels[8] = "a dog";
//        labels[7] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(label, 0, labels[0]);
        dataLoader.loadLabel_offset(label, 1, labels[1]);
        dataLoader.loadLabel_offset(label, 2, labels[2]);
        dataLoader.loadLabel_offset(label, 3, labels[3]);
        dataLoader.loadLabel_offset(label, 4, labels[4]);
        dataLoader.loadLabel_offset(label, 5, labels[5]);
        dataLoader.loadLabel_offset(label, 6, labels[6]);
        dataLoader.loadLabel_offset(label, 7, labels[7]);
//        dataLoader.loadLabel_offset(label, 8, labels[8]);
//        dataLoader.loadLabel_offset(label, 9, labels[9]);
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
        }
        
        network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
        }
        
//        Tensor noise1 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
//        GPUOP.getInstance().cudaRandn(noise1);
        
        for(int i = 0;i<10;i++) {
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
//            icplan.setTimestep_shift(i);
            
            Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 1.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\mmdit_4\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 5.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\mmdit_4\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_repa_cfg() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        int z_idx = 4;
        
        float y_prob = 0.1f;
        
        FluxDiT_REPA network = new FluxDiT_REPA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, z_idx, 768, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_24.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.57625645f,0.09338784f,-0.42964765f,-0.37670407f,-0.08795908f,-1.5591346f,-0.44817135f,-0.30657783f,0.13237366f,0.65041465f,-0.7171756f,1.9741942f,0.028869499f,0.6699318f,0.42782572f,1.7180899f,-1.1657567f,1.8240571f,-0.27507848f,-0.64780587f,0.57183015f,-0.015657624f,-1.6581186f,0.64075494f,2.8023534f,2.1147559f,-0.46823075f,0.06350619f,-1.3322849f,-0.3576193f,0.589975f,-0.28093442f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.092015f,4.054383f,3.416052f,3.608325f,3.6037152f,3.1869633f,3.0185716f,3.4761536f,3.730099f,3.6259832f,3.608243f,3.562f,3.7766607f,3.6814737f,3.208086f,3.5293725f,3.8835256f,3.5182753f,3.679344f,3.277554f,3.3914754f,3.0021727f,3.5673943f,4.2805233f,2.5594256f,3.3488555f,4.982151f,3.1698997f,3.1933618f,3.8946505f,4.275168f,4.1173005f}, true);

//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
        }
        
        network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
        }
        for(int i = 0;i<10;i++) {
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 1.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_repa\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 4.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_repa\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_sprint_cfg() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        FluxDiT_SPRINT network = new FluxDiT_SPRINT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt\\flux_sprint_b1_80.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.57625645f,0.09338784f,-0.42964765f,-0.37670407f,-0.08795908f,-1.5591346f,-0.44817135f,-0.30657783f,0.13237366f,0.65041465f,-0.7171756f,1.9741942f,0.028869499f,0.6699318f,0.42782572f,1.7180899f,-1.1657567f,1.8240571f,-0.27507848f,-0.64780587f,0.57183015f,-0.015657624f,-1.6581186f,0.64075494f,2.8023534f,2.1147559f,-0.46823075f,0.06350619f,-1.3322849f,-0.3576193f,0.589975f,-0.28093442f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.092015f,4.054383f,3.416052f,3.608325f,3.6037152f,3.1869633f,3.0185716f,3.4761536f,3.730099f,3.6259832f,3.608243f,3.562f,3.7766607f,3.6814737f,3.208086f,3.5293725f,3.8835256f,3.5182753f,3.679344f,3.277554f,3.3914754f,3.0021727f,3.5673943f,4.2805233f,2.5594256f,3.3488555f,4.982151f,3.1698997f,3.1933618f,3.8946505f,4.275168f,4.1173005f}, true);

//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
        }
        
        network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
        }
        for(int i = 0;i<10;i++) {
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 1.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 4.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_sprint_path_drop_cfg() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        FluxDiT_SPRINT network = new FluxDiT_SPRINT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt\\flux_sprint_b1_36.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
       
//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        }
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
        }
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky.";
                labels[2] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Half human, half robot, repaired human";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A car made out of vegetables.";
                labels[9] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
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
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_sprint_path_drop_cfg2() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        FluxDiT_SPRINT network = new FluxDiT_SPRINT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\dit_vavae\\flux\\sprint\\flux_sprint_b1_10.model";
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
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
//        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.57625645f,0.09338784f,-0.42964765f,-0.37670407f,-0.08795908f,-1.5591346f,-0.44817135f,-0.30657783f,0.13237366f,0.65041465f,-0.7171756f,1.9741942f,0.028869499f,0.6699318f,0.42782572f,1.7180899f,-1.1657567f,1.8240571f,-0.27507848f,-0.64780587f,0.57183015f,-0.015657624f,-1.6581186f,0.64075494f,2.8023534f,2.1147559f,-0.46823075f,0.06350619f,-1.3322849f,-0.3576193f,0.589975f,-0.28093442f}, true);
//        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.092015f,4.054383f,3.416052f,3.608325f,3.6037152f,3.1869633f,3.0185716f,3.4761536f,3.730099f,3.6259832f,3.608243f,3.562f,3.7766607f,3.6814737f,3.208086f,3.5293725f,3.8835256f,3.5182753f,3.679344f,3.277554f,3.3914754f,3.0021727f,3.5673943f,4.2805233f,2.5594256f,3.3488555f,4.982151f,3.1698997f,3.1933618f,3.8946505f,4.275168f,4.1173005f}, true);

//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        }
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
        }
        for(int i = 0;i<10;i++) {
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky.";
                labels[2] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Half human, half robot, repaired human";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A car made out of vegetables.";
                labels[9] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
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
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_sprint2_path_drop_cfg() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt2\\flux_sprint_b1_24_bak.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
       
//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        }
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
        }
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky.";
                labels[2] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Half human, half robot, repaired human";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A car made out of vegetables.";
                labels[9] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
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
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
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
	
	public static void test_flux_sprint2_path_drop_cfg_512() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
//        String model_path = "D:\\models\\dit_txt2_512\\flux_sprint_b1_512_2_5000.model";
        String model_path = "D:\\test\\dit_vavae\\models\\512\\flux_sprint_b1_512_0_80000.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
       
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
        labels[11] = "A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky.";
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
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
                }
            }

        	for(int it = 0;it<5;it++) {
            	
            	System.out.println("start create test images.");

                GPUOP.getInstance().cudaRandn(noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);
                
                icplan.latend_un_norm(sample, latendMean, latendStd);

                Tensor result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint_512\\"+i+"_" + it, result, mean, std);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);
                
                icplan.latend_un_norm(sample, latendMean, latendStd);

                result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint_512\\"+i+"_" + it + "_T", result, mean, std);
                
                System.out.println("finish create.");
            }
        	
        }

	}
	
	public static void test_flux_sprint2_path_drop_cfg2() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\dit_vavae\\models\\flux_sprint_b1_10.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
       
//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
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
            	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
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
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux_sprint2_path_drop_cfg3() throws Exception {
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
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
        
        FluxDiT_SPRINT3 network = new FluxDiT_SPRINT3(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt3\\flux_sprint_b1_40.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
       
//        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
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
        }
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, 0, 1);
        }
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky.";
                labels[2] = "A Ukiyoe-style painting, an astronaut riding a unicorn, In the background there is an ancient Japanese architecture.";
                labels[3] = "Shattered blue-and-white porcelain girl's face. fine texture. surreal";
                labels[4] = "Half human, half robot, repaired human";
                labels[5] = "Poster of a mechanical cat, techical Schematics viewed from front.";
                labels[6] = "A beautiful girl with golden hair, cool and sunny";
                labels[7] = "A Japanese girl walking along a path, surrounded by blooming oriental cherries, pink petals slowly falling down to the ground.";
                labels[8] = "A car made out of vegetables.";
                labels[9] = "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.";
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
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-3\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-3\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_flux2_cfg() throws Exception {
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

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 16;
        int en_depth = 2;
        int de_depth = 10;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        FluxDiT2 network = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt\\flux_ddt_b1_ema96.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
 	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
 	    ModeLoaderlUtils.loadData(noise, zdatas, "x");
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "a woman";
        labels[4] = "fruit cream cake";
        labels[5] = "bright red phlox flowers bloom in a garden";
//        labels[6] = "the cambridge shoulder bag";
//        labels[7] = "A yellow mushroom grows in the forest";
//        labels[8] = "a dog";
//        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(label, 0, labels[0]);
        dataLoader.loadLabel_offset(label, 1, labels[1]);
        dataLoader.loadLabel_offset(label, 2, labels[2]);
        dataLoader.loadLabel_offset(label, 3, labels[3]);
        dataLoader.loadLabel_offset(label, 4, labels[4]);
        dataLoader.loadLabel_offset(label, 5, labels[5]);
//        dataLoader.loadLabel_offset(label, 6, labels[6]);
//        dataLoader.loadLabel_offset(label, 7, labels[7]);
//        dataLoader.loadLabel_offset(label, 8, labels[8]);
//        dataLoader.loadLabel_offset(label, 9, labels[9]);
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
        }
        
        network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
        
        Tensor y_null = network.main.labelEmbd.getY_embedding();
        int part_input_size = y_null.dataLength;
        for(int b = 0;b<batchSize;b++) {
        	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
        }
        for(int i = 0;i<1;i++) {
        	System.out.println("start create test images.");
//            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 10.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_b1_ema\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
//            sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 10.5f);
//            
//            icplan.latend_un_norm(sample, latendMean, latendStd);
//
//            result = vae.decode(sample);
//            
//            JCuda.cudaDeviceSynchronize();
//            
//            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
//
//            showImgs("D:\\test\\dit_vavae\\flux_b1\\" + i + "_T", result, mean, std);
//            
//            System.out.println("finish create.");
        }
        
	}
	
	public static void test_ema() throws Exception {
	       // ========== 路径配置 ==========
	       String vocabPath = "c:\\temp\\vocab.json";
	       String mergesPath = "c:\\temp\\merges.txt";
	       String clipWeight = "c:\\temp\\CLIP-GmP-ViT-L-14.json";
	       String vaeWeight = "c:\\temp\\vavae.json";
	       String model_path = "c:\\temp\\flux_ddt_b1_96.model";
	       String ema_model_path = "c:\\temp\\flux_ddt_b1_ema96.model";
	       String outputPath = "c:\\temp\\ema\\";
	       float cfgScale = 1.0f;

	       // ========== 模型参数配置 ==========
	       int imgSize = 256;
	       int maxContextLen = 77;
	       int batchSize = 10;
	       float[] mean = new float[]{0.5f, 0.5f, 0.5f};
	       float[] std = new float[]{0.5f, 0.5f, 0.5f};

	       // ========== 初始化组件 ==========
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
	       ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

	       int latendDim = 32;
	       int num_res_blocks = 2;
	       int[] ch_mult = new int[]{1, 1, 2, 2, 4};
	       int ch = 128;
	       VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
	       vae.CUDNN = true;
	       vae.learnRate = 0.001f;
	       vae.RUN_MODEL = RunModel.EVAL;
	       ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);

	       int ditHeadNum = 12;
	       int latendSize = 16;
	       int en_depth = 2;
	       int de_depth = 10;
	       int timeSteps = 1000;
	       int mlpRatio = 4;
	       int patchSize = 1;
	       int hiddenSize = 768;

	       float y_prob = 0.1f;

	       // 保存噪声数据用于EMA模型
	       Tensor[] savedNoises = new Tensor[10];
	       
	       // ========== 第一阶段：普通模型测试 ==========
	       System.out.println("=== 第一阶段：普通模型测试 ===");
	       
	       // 创建普通模型
	       FluxDiT2 network = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
	       network.CUDNN = true;
	       network.learnRate = 0.0002f;
	       network.RUN_MODEL = RunModel.EVAL;

	       ICPlan icplan = new ICPlan(network.tensorOP);

	       // 加载普通模型权重
	       ModelUtils.loadModel(network, model_path);

	       Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);

	       Tensor condInput = null;
	       Tensor condInput_ynull = null;
	       Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);

	       Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
	       Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
	       Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);

	       Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
	       Tensor cos = cs[0];
	       Tensor sin = cs[1];

	       Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
	       Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

	       String[] labels = new String[10];
	       labels[0] = "A cat";
	       labels[1] = "a vibrant anime mountain lands";
	       labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
	       labels[3] = "a woman";
	       labels[4] = "fruit cream cake";
	       labels[5] = "bright red phlox flowers bloom in a garden";
	       labels[6] = "the cambridge shoulder bag";
	       labels[7] = "A yellow mushroom grows in the forest";
	       labels[8] = "a dog";
	       labels[9] = "A lovely corgi is taking a walk under the sea";
	       
	       // 运行普通模型的所有测试
	       for(int i = 0;i<10;i++) {
	           System.out.println("Normal model - generating image " + i);
	           
	           // 直接使用BPE tokenizer处理文本
	           for(int j = 0; j < batchSize; j++) {
	               int[] tokens = bpe.encodeInt(labels[j], maxContextLen);
	               for(int k = 0; k < maxContextLen; k++) {
	                   if(k < tokens.length) {
	                       label.data[j * maxContextLen + k] = tokens[k];
	                   } else {
	                       label.data[j * maxContextLen + k] = 0;
	                   }
	               }
	           }
	           label.hostToDevice();
	           
	           condInput = clip.get_full_clip_prompt_embeds(label);

	           if(condInput_ynull == null) {
	               condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
	           }

	           network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);

	           Tensor y_null = network.main.labelEmbd.getY_embedding();
	           int part_input_size = y_null.dataLength;
	           for(int b = 0;b<batchSize;b++) {
	               network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
	           }

	           // 生成噪声并保存
	           GPUOP.getInstance().cudaRandn(noise);
	           savedNoises[i] = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
	           noise.copyGPU(savedNoises[i]);

	           // 使用普通模型生成图像
	           Tensor sample_normal = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, cfgScale);
	           icplan.latend_un_norm(sample_normal, latendMean, latendStd);
	           Tensor result_normal = vae.decode(sample_normal);
	           JCuda.cudaDeviceSynchronize();
	           result_normal.data = MatrixOperation.clampSelf(result_normal.syncHost(), -1, 1);
	           showImgs(outputPath + i + "_normal", result_normal, mean, std);

	           System.out.println("Normal model - finished image " + i);
	       }
	       
	       // 释放普通模型内存
	       System.out.println("=== 释放普通模型内存 ===");
	       network = null;
	       System.gc();
	       
	       // ========== 第二阶段：EMA模型测试 ==========
	       System.out.println("=== 第二阶段：EMA模型测试 ===");
	       
	       // 创建EMA模型
	       FluxDiT2 network_ema = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
	       network_ema.CUDNN = true;
	       network_ema.learnRate = 0.0002f;
	       network_ema.RUN_MODEL = RunModel.EVAL;
	       
	       // 加载EMA模型权重
	       ModelUtils.loadModel(network_ema, ema_model_path);
	       
	       // 运行EMA模型的所有测试（使用保存的噪声）
	       for(int i = 0;i<10;i++) {
	           System.out.println("EMA model - generating image " + i);
	           
	           // 直接使用BPE tokenizer处理文本
	           for(int j = 0; j < batchSize; j++) {
	               int[] tokens = bpe.encodeInt(labels[j], maxContextLen);
	               for(int k = 0; k < maxContextLen; k++) {
	                   if(k < tokens.length) {
	                       label.data[j * maxContextLen + k] = tokens[k];
	                   } else {
	                       label.data[j * maxContextLen + k] = 0;
	                   }
	               }
	           }
	           label.hostToDevice();
	           
	           condInput = clip.get_full_clip_prompt_embeds(label);

	           if(condInput_ynull == null) {
	               condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
	           }

	           network_ema.tensorOP.cat_batch(condInput, condInput, condInput_ynull);

	           Tensor y_null = network_ema.main.labelEmbd.getY_embedding();
	           int part_input_size = y_null.dataLength;
	           for(int b = 0;b<batchSize;b++) {
	               network_ema.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
	           }

	           // 使用EMA模型生成图像（使用保存的相同噪声）
	           Tensor sample_ema = icplan.forward_with_cfg(network_ema, savedNoises[i], t, condInput_ynull, cos, sin, latend, eps, cfgScale);
	           icplan.latend_un_norm(sample_ema, latendMean, latendStd);
	           Tensor result_ema = vae.decode(sample_ema);
	           JCuda.cudaDeviceSynchronize();
	           result_ema.data = MatrixOperation.clampSelf(result_ema.syncHost(), -1, 1);
	           showImgs(outputPath + i + "_ema", result_ema, mean, std);

	           System.out.println("EMA model - finished image " + i);
	       }
	       
	       System.out.println("=== 测试完成 ===");
	}
	
	public static void test_flux3_cfg() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 8;
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
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 16;
        int latendSize = 16;
        int en_depth = 4;
        int de_depth = 20;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1024;
        
        float y_prob = 0.1f;
        
        FluxDiT2 network = new FluxDiT2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, en_depth, de_depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\dit_vavae\\flux\\flux2_dit_l1_0_80000.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[10];
        for(int i = 0;i<10;i++) {
        	System.out.println("start create test images.");
        	labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
            labels[3] = "the cambridge shoulder bag";
            labels[4] = "fruit cream cake";
            labels[5] = "a woman";
	        labels[6] = "A panda sleep on the water";
            labels[7] = "a dog";
//            labels[8] = "a dog";
//            labels[9] = "A lovely corgi is taking a walk under the sea";
            dataLoader.loadLabel_offset(label, 0, labels[0]);
            dataLoader.loadLabel_offset(label, 1, labels[1]);
            dataLoader.loadLabel_offset(label, 2, labels[2]);
            dataLoader.loadLabel_offset(label, 3, labels[3]);
            dataLoader.loadLabel_offset(label, 4, labels[4]);
            dataLoader.loadLabel_offset(label, 5, labels[5]);
            dataLoader.loadLabel_offset(label, 6, labels[6]);
            dataLoader.loadLabel_offset(label, 7, labels[7]);
//            dataLoader.loadLabel_offset(label, 8, labels[8]);
//            dataLoader.loadLabel_offset(label, 9, labels[9]);
            condInput = clip.get_full_clip_prompt_embeds(label);
            
            if(condInput_ynull == null) {
            	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
            }
            
            network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
            
            Tensor y_null = network.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
            }
            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 1.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_l1\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 10.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_l1\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_dinov2() throws Exception {
		
//		String labelPath = "D:\\dataset\\amine\\data.json";
//        String imgDirPath = "D:\\dataset\\images_224_224\\";
//        boolean horizontalFilp = false;
        int imgSize = 448;
//        int maxContextLen = 77;
//        int batchSize = 1;
        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
//        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
//        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
//        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
//        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
		
        int patchSize = 14;
        int hiddenSize = 768;
        int headNum = 12;
        int depth = 12;
        int mlpRatio = 4;
        Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, patchSize, hiddenSize, headNum, depth, mlpRatio);
        dinov.CUDNN = true;
        dinov.RUN_MODEL = RunModel.EVAL;

        String weight = "D:\\models\\dionv2-14-b-512.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), dinov, depth, true);
        
        String save_model_path = "D:\\models\\dionv2-14-b-512.model";
        ModelUtils.saveModel(dinov, save_model_path);
      
//        String model_path = "D:\\models\\dionv2-14-b.model";
//        ModelUtils.loadModel(dinov, model_path);
//        
//        String imgPath = "D:\\dataset\\images_224_224\\dalle3_1m_00000008.jpg";
//        
//        float[] data = YoloImageUtils.loadImgDataToArray(imgPath, true, mean, std);
//        
//        Tensor img = new Tensor(1, 3, 224, 224, data, true);
//        
//        img.showDM();
//        
//        Tensor z =  dinov.forward_features(img);
//        
//        z.showDM("z");
        
    }
	
    public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }
    
    public static void loadWeight(Map<String, Object> weightMap, Dinov2 dit, int depth, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.cls_token, weightMap, 3, "cls_token");
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.pos_embed, weightMap, 3, "pos_embed"); 
        
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.patchEmbedding.weight, weightMap, "patch_embed.proj.weight");
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.patchEmbedding.bias, weightMap, "patch_embed.proj.bias");
    
        for(int i = 0;i<depth;i++) {
        	NestedTensorBlock block = dit.main.blocks.get(i);
        	
        	block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
            block.norm1.beta = ModeLoaderlUtils.loadData(block.norm1.beta, weightMap, 1, "blocks."+i+".norm1.bias");
            
            ModeLoaderlUtils.loadData(block.attn.qkvLinerLayer.weight, weightMap, "blocks."+i+".attn.qkv.weight");
            ModeLoaderlUtils.loadData(block.attn.qkvLinerLayer.bias, weightMap, "blocks."+i+".attn.qkv.bias");
        	ModeLoaderlUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
            ModeLoaderlUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
            
            block.scale1.weight = ModeLoaderlUtils.loadData(block.scale1.weight, weightMap, 1, "blocks."+i+".ls1.gamma");
            
            block.norm2.gamma = ModeLoaderlUtils.loadData(block.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight"); 
            block.norm2.beta = ModeLoaderlUtils.loadData(block.norm2.beta, weightMap, 1, "blocks."+i+".norm2.bias");
            
            ModeLoaderlUtils.loadData(block.mlp.linear1.weight, weightMap, "blocks."+i+".mlp.fc1.weight");
        	ModeLoaderlUtils.loadData(block.mlp.linear1.bias, weightMap, "blocks."+i+".mlp.fc1.bias");
        	ModeLoaderlUtils.loadData(block.mlp.linear2.weight, weightMap, "blocks."+i+".mlp.fc2.weight");
        	ModeLoaderlUtils.loadData(block.mlp.linear2.bias, weightMap, "blocks."+i+".mlp.fc2.bias");
        	
        	block.scale2.weight = ModeLoaderlUtils.loadData(block.scale2.weight, weightMap, 1, "blocks."+i+".ls2.gamma");
        }
        
        dit.main.norm.gamma = ModeLoaderlUtils.loadData(dit.main.norm.gamma, weightMap, 1, "norm.weight"); 
        dit.main.norm.beta = ModeLoaderlUtils.loadData(dit.main.norm.beta, weightMap, 1, "norm.bias");
        
    }
    
    public static void loadWeight(Map<String, Object> weightMap, FluxDiT3 dit, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
//        block.context_block.norm1.gamma = ModeLoaderlUtils.loadData(block.context_block.norm1.gamma, weightMap, 1, "context_block.norm1.weight"); 
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight", 4);
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");

        ModeLoaderlUtils.loadData(dit.main.labelEmbd.init_y_embedding(), weightMap, "y_embedder.y_embedding");
        
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");

        for(int i = 0;i<28;i++) {
        	
        	FluxDiTBlock2 block = dit.main.blocks.get(i);
        	
        	block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
        	block.norm3.gamma = ModeLoaderlUtils.loadData(block.norm3.gamma, weightMap, 1, "blocks."+i+".norm2.weight");  
        	
        	ModeLoaderlUtils.loadData(block.attn.qkvLinerLayer.weight, weightMap, "blocks."+i+".attn.qkv.weight");
            ModeLoaderlUtils.loadData(block.attn.qkvLinerLayer.bias, weightMap, "blocks."+i+".attn.qkv.bias");
        	ModeLoaderlUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
            ModeLoaderlUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
            
        	ModeLoaderlUtils.loadData(block.mlp.w12.weight, weightMap, "blocks."+i+".mlp.w12.weight");
        	ModeLoaderlUtils.loadData(block.mlp.w12.bias, weightMap, "blocks."+i+".mlp.w12.bias");
        	ModeLoaderlUtils.loadData(block.mlp.w3.weight, weightMap, "blocks."+i+".mlp.w3.weight");
        	ModeLoaderlUtils.loadData(block.mlp.w3.bias, weightMap, "blocks."+i+".mlp.w3.bias");
        	
        	ModeLoaderlUtils.loadData(block.adaLN_modulation.weight, weightMap, "blocks."+i+".adaLN_modulation.1.weight");
        	ModeLoaderlUtils.loadData(block.adaLN_modulation.bias, weightMap, "blocks."+i+".adaLN_modulation.1.bias");
        }
        
        dit.main.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(dit.main.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight"); 
        ModeLoaderlUtils.loadData(dit.main.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear.weight, weightMap, "final_layer.adaLN_modulation.1.weight");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear.bias, weightMap, "final_layer.adaLN_modulation.1.bias");
    }
    
//    public static void test() {
//    	 
//    	 int N = 2;
//    	
//    	 int textEmbedDim = 768;
//    	 int maxContext = 77;
//    	 
//    	 int latendDim = 32;
//    	 int ditHeadNum = 6;
//         int latendSize = 16;
//         int depth = 6;
//         int timeSteps = 1000;
//         int mlpRatio = 4;
//         int patchSize = 1;
//         int hiddenSize = 384;
//         
//         float y_prob = 0.0f;
//         
//         FluxDiT dit = new FluxDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
//         dit.CUDNN = true;
//         dit.learnRate = 0.0002f;
//         
//        String weight = "D:\\models\\test\\dit.json";
//        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), dit, true);
//         
//     	String inputPath = "D:\\models\\test\\dit_x.json";
//        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//        Tensor img = new Tensor(N, latendDim, latendSize, latendSize, true);
//        ModeLoaderlUtils.loadData(img, datas, "x");
//        
//     	String cyPath = "D:\\models\\test\\dit_cy.json";
//        Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
//        Tensor cy = new Tensor(N * maxContext, 1, 1, textEmbedDim, true);
//        ModeLoaderlUtils.loadData(cy, cydatas, "cy", 2);
//
////     	String deltaPath = "D:\\models\\test\\dit_delta.json";
////        Map<String, Object> deltadatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
////        Tensor delta = new Tensor(N, latendDim, latendSize, latendSize, true);
////        ModeLoaderlUtils.loadData(delta, deltadatas, "delta");
//        
//     	String noicePath = "D:\\models\\test\\dit_noise.json";
//        Map<String, Object> ndatas = LagJsonReader.readJsonFileSmallWeight(noicePath);
//        Tensor noice = new Tensor(N, latendDim, latendSize, latendSize, true);
//        ModeLoaderlUtils.loadData(noice, ndatas, "noise");
//        
//        Tensor[] cs = RoPEKernel.getCosAndSin2D(dit.time, dit.hiddenSize, dit.headNum);
//        Tensor cos = cs[0];
//        Tensor sin = cs[1];
//        
//        Tensor t = new Tensor(N, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
//        cy.showDM("cy");
//        
//        ICPlan icplan = new ICPlan(dit.tensorOP);
//        
//        Tensor xt = new Tensor(N, latendDim, latendSize, latendSize, true);
//        Tensor ut = new Tensor(N, latendDim, latendSize, latendSize, true);
//        
//        Tensor cosine_similarity_loss = new Tensor(N, latendDim, latendSize, latendSize, true);
//        
//        for(int i = 0;i<1000;i++) {
//        	
//        	/**
//             * latend add noise
//             */
//            icplan.compute_xt(t, noice, img, xt);
//            icplan.compute_ut(t, noice, img, ut);
//        	
//        	 Tensor output = dit.forward(xt, t, cy, cos, sin);
//             
//        	 /**
//              * loss
//              */
//             Tensor loss = dit.loss(output, ut);
//          
//             /**
//              * loss diff
//              */
//             Tensor lossDiff = dit.lossDiff(output, ut);
////             lossDiff.showDM();
////             icplan.cosine_similarity_loss(output, ut, cosine_similarity_loss);
//             
////             icplan.cosine_similarity_loss_back(output, ut, cosine_similarity_loss);
////             
////             dit.tensorOP.add(lossDiff, cosine_similarity_loss, lossDiff);
//             
//             /**
//              * back
//              */
//             dit.back(lossDiff, cos, sin);
//             
//             /**
//              * update
//              */
//             dit.update();
//             JCudaDriver.cuCtxSynchronize();
//             
//             float mse_loss = MatrixOperation.sum(loss.syncHost()) / N;
//             System.err.println(i+"--" + mse_loss);
//        }
//
//    }
    
    public static void test2() {
   	 
	   	int batchSize = 6;
	   
	   	int textEmbedDim = 768;
	   	int maxContext = 77;
	   	
	   	String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
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
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        int latendDim = 32;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);

	   	int ditHeadNum = 16;
        int latendSize = 16;
        int depth = 28;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        
        float y_prob = 0.1f;
        
        FluxDiT3 network = new FluxDiT3(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
//       String weight = "D:\\test\\dit_vavae\\lightingdit.json";
//       loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), network, true);
//        
//       String save_model_path = "D:\\test\\dit_vavae\\flux_xl1.model";
//       ModelUtils.saveModel(network, save_model_path);
       
       String model_path = "D:\\test\\dit_vavae\\flux_xl1.model";
       ModelUtils.loadModel(network, model_path);
       
       ICPlan icplan = new ICPlan(network.tensorOP);
       
       Tensor label = new Tensor(batchSize * maxContext, 1, 1, 1, true);
       
       Tensor condInput = null;
       Tensor condInput_ynull = null;
       Tensor t = new Tensor(batchSize * 2, 1, 1, 1, true);
//       Tensor t1 = new Tensor(batchSize, 1, 1, 1, true);
       
       Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
       Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
       Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
       
       Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
       
       Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
       Tensor cos = cs[0];
       Tensor sin = cs[1];
       
       Tensor latendMean = new Tensor(1, latendDim, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
       Tensor latendStd = new Tensor(1, latendDim, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

       String cyPath = "D:\\test\\dit_vavae\\lightingdit_ms.json";
	   Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	   ModeLoaderlUtils.loadData(latendMean, cydatas, "mean", 4);
	   ModeLoaderlUtils.loadData(latendStd, cydatas, "std", 4);
       
	   latendMean.view(latendDim, 1, 1, 1);
	   latendStd.view(latendDim, 1, 1, 1);
	   
//       String zPath = "D:\\test\\dit_vavae\\lightingdit_z.json";
//	   Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//	   ModeLoaderlUtils.loadData(noise, zdatas, "x");
	   
       network.RUN_MODEL = RunModel.EVAL;
       String[] labels = new String[batchSize];
       labels[0] = "A cat holding a sign that says hello world";
       labels[1] = "a vibrant anime mountain lands";
       labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
       labels[3] = "the cambridge shoulder bag";
       labels[4] = "fruit cream cake";
       labels[5] = "a woman";
//       labels[6] = "A panda sleep on the water";
//       labels[7] = "a dog";

       dataLoader.loadLabel_offset(label, 0, labels[0]);
       dataLoader.loadLabel_offset(label, 1, labels[1]);
       dataLoader.loadLabel_offset(label, 2, labels[2]);
       dataLoader.loadLabel_offset(label, 3, labels[3]);
       dataLoader.loadLabel_offset(label, 4, labels[4]);
       dataLoader.loadLabel_offset(label, 5, labels[5]);
//       dataLoader.loadLabel_offset(label, 6, labels[6]);
//       dataLoader.loadLabel_offset(label, 7, labels[7]);

       condInput = clip.get_full_clip_prompt_embeds(label);
       
       if(condInput_ynull == null) {
       	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number * 2, condInput.channel, condInput.height, condInput.width, true);
       }
       
       network.tensorOP.cat_batch(condInput, condInput, condInput_ynull);
       
       Tensor y_null = network.main.labelEmbd.getY_embedding();
       int part_input_size = y_null.dataLength;
       for(int b = 0;b<batchSize;b++) {
    	   network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, (batchSize + b) * part_input_size, 1);
       }
       for(int i = 0;i<10;i++) {
    	   System.out.println("start create test images.");
           GPUOP.getInstance().cudaRandn(noise);
           noise.copyGPU(noise2);
           
           Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput_ynull, cos, sin, latend, eps, 1.0f);
           
           icplan.latend_un_norm(sample, latendMean, latendStd);

           Tensor result = vae.decode(sample);
           
           JCuda.cudaDeviceSynchronize();
           
           result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

           showImgs("D:\\test\\dit_vavae\\flux_l1_test\\" + i, result, mean, std);
           
           System.out.println("finish create.");
           
           sample = icplan.forward_with_cfg(network, noise2, t, condInput_ynull, cos, sin, latend, eps, 10.0f);
           
           icplan.latend_un_norm(sample, latendMean, latendStd);

           result = vae.decode(sample);
           
           JCuda.cudaDeviceSynchronize();
           
           result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

           showImgs("D:\\test\\dit_vavae\\flux_l1_test\\" + i + "_T", result, mean, std);
           
           System.out.println("finish create.");
       }
    }
    
    public static void test_random() {
        // Test 4: Basic structured mask with 2x2 groups, select 1 per group
        System.out.println("Test 4: Structured mask (2x2 groups, select 1)");
        int B = 32;
        int tokensH = 16;
        int tokensW = 16;
        int groupSize = 2;
        int selectionPerGroup = 1;

        System.out.println("B=" + B + ", H=" + tokensH + ", W=" + tokensW);
        System.out.println("groupSize=" + groupSize + ", selectionPerGroup=" + selectionPerGroup);

        long startTime = System.currentTimeMillis();
        float[] result = RandomMaskUtils.getStructuredMaskWithRandomOffset_idskeep(B, tokensH, tokensW, groupSize, selectionPerGroup);
        long endTime = System.currentTimeMillis();
        
        System.out.println("\nTime: " + (endTime - startTime) + "ms");
        System.err.println(JsonUtils.toJson(result));
    }
    
    public static void main(String[] args) {
		 
	        try {
	           
//	        	flux_dit_b2_iddpm_train();
	        	
//	        	flux_dit_b2_iddpm_train_unsample();

//	        	test_flux_cfg();
	        	
//	        	test();
	        	
//	        	test2();
	        	
//	        	test_flux2_cfg();

//	        	test_ema();
	        	
//	        	test_flux3_cfg();
	        	
//	        	test_dinov2();
	        	
//	        	flux_repa_b1_iddpm_train();
	        	
//	        	flux_reg_b1_iddpm_train();
	        	
//	        	test_flux_repa_cfg();
	        	
//	        	flux_sprint_b1_iddpm_train();
	        	
//	        	flux_sprint_b1_iddpm_train2();
//	        	
//	        	flux_sprint_b1_iddpm_train2_512();
	        	
//	        	test_flux_sprint2_path_drop_cfg_512();
	        	
	        	flux_sprint_b1_iddpm_train3();
	        	
//	        	flux_sprint_b1_iddpm_train4();
	        	
//	        	test_flux_sprint2_path_drop_cfg();
	        	
//	        	test_flux_sprint2_path_drop_cfg2();
	        	
//	        	test_flux_sprint2_path_drop_cfg3();
	        	
//	        	test_flux_sprint_cfg();
	        	
//	        	test_flux_sprint_path_drop_cfg();
	        	
//	        	test_flux_sprint_path_drop_cfg2();
	        	
//	        	flux_sprint_reg_b1_iddpm_train();
	        	
	        } catch (Exception e) {
	            // TODO: handle exception
	            e.printStackTrace();
	        } finally {
	            // TODO: handle finally clause
	            CUDAMemoryManager.free();
	        }
	  }
	
}
