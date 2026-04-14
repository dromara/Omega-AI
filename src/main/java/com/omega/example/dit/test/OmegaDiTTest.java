package com.omega.example.dit.test;

import java.io.IOException;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.T5Encoder;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.OmegaDiT;
import com.omega.engine.nn.network.dit.OmegaDiTFullLabel;
import com.omega.engine.nn.network.vae.Flux_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.dataset.LatendDataset_t5_clip;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.sd.utils.SDImageLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.runtime.JCuda;

public class OmegaDiTTest {

	public static void omega_sprint_b1_iddpm_train() throws Exception {
//		String dataPath = "D:\\dataset\\flux_train_sampled\\vavae_latend.bin";
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
		String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
//        String clipDataPath = "D:\\dataset\\amine\\vavae_t5.bin";
		
        int batchSize = 20;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
//        String labelPath = "D:\\dataset\\flux_train_sampled\\metadata.json";
//		String imgDirPath = "D:\\dataset\\flux_train_sampled\\images_224\\";
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
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt2\\flux_sprint_b1_32.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
//        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.68802416f,0.23106125f,0.34798032f,-0.11941326f,-0.964228f,-1.2219781f,-0.13273178f,0.39500353f,1.1611824f,0.33286065f,-0.9072468f,1.1964253f,0.77751267f,1.4342179f,-0.27347723f,0.6652678f,-1.0584584f,0.39614016f,0.10801358f,-1.3052012f,0.13688391f,-0.15965684f,-0.6702366f,0.68603706f,2.3822904f,1.4025455f,-0.81627595f,-0.32144645f,-0.109326765f,-1.3960866f,-0.38953456f,0.46669957f}, true);
//        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.117937f,4.2716165f,3.3511283f,3.7278311f,3.8668838f,3.888655f,2.970441f,3.5744536f,3.8475676f,3.9073176f,3.7287266f,3.8889313f,3.85656f,3.6298664f,3.286395f,3.3439715f,4.0913005f,3.8466156f,3.4434404f,3.1861749f,3.2535207f,3.0563624f,3.8309762f,4.52857f,2.9302385f,3.4940221f,4.7427244f,3.41022f,3.4938436f,3.7268739f,4.6008964f,4.3314376f}, true);
        
//        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
//        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
        //[0.2580713,0.41303077,-0.17897944,-0.593331,-0.18632421,-1.7637733,-0.54968524,0.027559364,0.41386542,0.62200934,-1.0693398,1.680629,0.103141084,0.8374718,0.18392913,0.8740971,-1.0440301,1.4362086,-0.32072797,-0.7607167,0.58934206,0.11546381,-1.2952809,0.612079,3.0053487,1.6832577,0.056598995,-0.2920768,-0.9245258,-0.65787494,0.56643915,-0.3693799]
        //[4.1784525,4.2560844,3.4338517,3.6914275,3.6218529,3.390063,3.060225,3.6699965,3.9034598,3.753831,3.7214353,3.6986103,3.716805,3.6442988,3.22384,3.3135345,4.127044,3.642752,3.6825655,3.0699682,3.3633635,2.9721425,3.8364065,4.5173664,2.760846,3.4888804,4.732718,3.2708733,3.353005,3.7254717,4.6618576,4.287393]
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.2580713f,0.41303077f,-0.17897944f,-0.593331f,-0.18632421f,-1.7637733f,-0.54968524f,0.027559364f,0.41386542f,0.62200934f,-1.0693398f,1.680629f,0.103141084f,0.8374718f,0.18392913f,0.8740971f,-1.0440301f,1.4362086f,-0.32072797f,-0.7607167f,0.58934206f,0.11546381f,-1.2952809f,0.612079f,3.0053487f,1.6832577f,0.056598995f,-0.2920768f,-0.9245258f,-0.65787494f,0.56643915f,-0.3693799f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1784525f,4.2560844f,3.4338517f,3.6914275f,3.6218529f,3.390063f,3.060225f,3.6699965f,3.9034598f,3.753831f,3.7214353f,3.6986103f,3.716805f,3.6442988f,3.22384f,3.3135345f,4.127044f,3.642752f,3.6825655f,3.0699682f,3.3633635f,2.9721425f,3.8364065f,4.5173664f,2.760846f,3.4888804f,4.732718f,3.2708733f,3.353005f,3.7254717f,4.6618576f,4.287393f}, true);
        
        optimizer.train_Flux_Sprint_ICPlan2(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_sprint_b1_iddpm_train_512() throws Exception {
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
        dit.learnRate = 5e-5f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt2_512\\flux_sprint_b1_512_3_10000.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        optimizer.train_Flux_Sprint_ICPlan2_512(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2_512//", latend_mean, latend_std, 1f, 1);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1_512.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_sprint_b1_iddpm_train_fluxvae() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_fluxvae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 30;
        int latendDim = 16;
        int height = 32;
        int width = 32;
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
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-7f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_20.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.0355735719203949f,0.0048736147582530975f,-0.003450946183875203f,-0.03965449333190918f,0.036058880388736725f,0.04668542370200157f,-0.05957012623548508f,-0.001957265892997384f,-0.025739729404449463f,0.008893698453903198f,0.016873272135853767f,-0.0028637342620640993f,0.020259900018572807f,-0.005238725338131189f,-0.024182798340916634f,-0.0040397001430392265f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {2.339289665222168f,2.355745315551758f,2.364927053451538f,2.34753680229187f,2.3591816425323486f,2.38031268119812f,2.348323345184326f,2.343754291534424f,2.363278865814209f,2.3746819496154785f,2.3765056133270264f,2.362192392349243f,2.3591220378875732f,2.3649189472198486f,2.345620632171631f,2.3334505558013916f}, true);

        optimizer.train_Flux_Sprint_ICPlan2(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt_flux//", latend_mean, latend_std, 0.3611f, 0.1159f, 4);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_sprint_b1_iddpm_train_fluxvae_512() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_fluxvae_latend_512.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		
        int batchSize = 8;
        int latendDim = 16;
        int height = 64;
        int width = 64;
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
        int latendSize = 64;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.75f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

//        String model_path = "D:\\models\\dit_txt_flux_512\\flux_sprint_b1_2.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 4, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.0355735719203949f,0.0048736147582530975f,-0.003450946183875203f,-0.03965449333190918f,0.036058880388736725f,0.04668542370200157f,-0.05957012623548508f,-0.001957265892997384f,-0.025739729404449463f,0.008893698453903198f,0.016873272135853767f,-0.0028637342620640993f,0.020259900018572807f,-0.005238725338131189f,-0.024182798340916634f,-0.0040397001430392265f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {2.339289665222168f,2.355745315551758f,2.364927053451538f,2.34753680229187f,2.3591816425323486f,2.38031268119812f,2.348323345184326f,2.343754291534424f,2.363278865814209f,2.3746819496154785f,2.3765056133270264f,2.362192392349243f,2.3591220378875732f,2.3649189472198486f,2.345620632171631f,2.3334505558013916f}, true);

        optimizer.train_Flux_Sprint_ICPlan2(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt_flux_512//", latend_mean, latend_std, 0.3611f, 0.1159f, 1);
        String save_model_path = "D://models//dit_txt_flux//fluxvae_sprint_b1_512.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void omega_full_b1_iddpm_train() throws Exception {
//		String dataPath = "D:\\dataset\\flux_train_sampled\\vavae_latend.bin";
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
		String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		String t5DataPath = "D:\\dataset\\amine\\vavae_t5.bin";
        
        int batchSize = 22;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int clip_len = 77;
        int t5_len = 120;
        int clipEmbedDim = 768;
        int t5EmbedDim = 2048;
        
        LatendDataset_t5_clip dataLoader = new LatendDataset_t5_clip(dataPath, clipDataPath, t5DataPath, batchSize, latendDim, height, width, clip_len, clipEmbedDim, t5_len, t5EmbedDim, BinDataType.float32);
        
//        String labelPath = "D:\\dataset\\flux_train_sampled\\metadata.json";
//		String imgDirPath = "D:\\dataset\\flux_train_sampled\\images_224\\";
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
        
        OmegaDiTFullLabel dit = new OmegaDiTFullLabel(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, clipEmbedDim, clip_len, t5EmbedDim, t5_len, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D://models//dit_txt2//flux_sprint_b1_12.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
//        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.68802416f,0.23106125f,0.34798032f,-0.11941326f,-0.964228f,-1.2219781f,-0.13273178f,0.39500353f,1.1611824f,0.33286065f,-0.9072468f,1.1964253f,0.77751267f,1.4342179f,-0.27347723f,0.6652678f,-1.0584584f,0.39614016f,0.10801358f,-1.3052012f,0.13688391f,-0.15965684f,-0.6702366f,0.68603706f,2.3822904f,1.4025455f,-0.81627595f,-0.32144645f,-0.109326765f,-1.3960866f,-0.38953456f,0.46669957f}, true);
//        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.117937f,4.2716165f,3.3511283f,3.7278311f,3.8668838f,3.888655f,2.970441f,3.5744536f,3.8475676f,3.9073176f,3.7287266f,3.8889313f,3.85656f,3.6298664f,3.286395f,3.3439715f,4.0913005f,3.8466156f,3.4434404f,3.1861749f,3.2535207f,3.0563624f,3.8309762f,4.52857f,2.9302385f,3.4940221f,4.7427244f,3.41022f,3.4938436f,3.7268739f,4.6008964f,4.3314376f}, true);
        
        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
        optimizer.train_Flux_Sprint_ICPlan(dinov, dataLoader2, dataLoader, icplan, "D://models//dit_txt2//", latend_mean, latend_std, 1f, 4);
        String save_model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void test_omega_sprint_path_drop_cfg() throws Exception {
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
	
	public static void test_omega_sprint_path_drop_cfg_512() throws Exception {
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
        
        String model_path = "D:\\models\\dit_txt_flux_512\\flux_sprint_b1_512.model";
//        String model_path = "D:\\models\\dit_txt\\flux_sprint_b1_512.model";
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
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);                }
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
	
    public static void loadLabel_offset(SentencePieceTokenizer tokenizer, Tensor label, Tensor mask, int index, String labelStr, int maxContextLen) {
    	int[] ids = tokenizer.encodeInt(labelStr, maxContextLen);
        for (int j = 0; j < maxContextLen; j++) {
        	int val = ids[j];
        	label.data[index * maxContextLen + j] = val;
        	if(val != tokenizer.pad()) {
        		mask.data[index * maxContextLen + j] = 0;
        	}else {
        		mask.data[index * maxContextLen + j] = -3.4028e+38f;
        	}
        }
    }
	
	public static void test_omega_t5_path_drop_cfg() throws Exception {

        int imgSize = 256;
        int maxContextLen = 120;
        int batchSize = 10;
        
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        
        String tokenizer_path = "D:\\models\\t5\\spiece.model";
		SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path);
       
        int time = maxContextLen;
		int voc_size = 250112;
		int num_layers = 24;
		int head_num = 32;
		int textEmbedDim = 2048;
		int d_ff = 5120;
		T5Encoder t5 = new T5Encoder(LossType.MSE, UpdaterType.adamw, voc_size, num_layers, head_num, time, textEmbedDim, d_ff, false);
		t5.CUDNN = true;
		t5.RUN_MODEL = RunModel.EVAL;
    	
		String t5_path = "D://models//t5//t5_encoder.model";
        ModelUtils.loadModel(t5, t5_path);

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
        
        String model_path = "D://models//dit_txt//flux_sprint_b1.model";
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

        Tensor mask = new Tensor(batchSize, 1, 1, maxContextLen, true);
        
        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
//        labels[0] = "一只猫";
//        labels[1] = "一只穿着西装的猪";
//        labels[2] = "赛博朋克风，跑车";
//        labels[3] = "年轻的女孩站在春季的火车站月台上。身穿灰色风衣，白色衬衫。";
//        labels[4] = "一个小女孩戴着一顶精致的花环，花环上装饰着六朵鲜艳的花朵——两朵白色的花，中间有黄色的花心；两朵柔和的粉红色玫瑰；以及两朵浅粉色的花。花环周围点缀着茂盛的绿叶。她穿着一件可爱的蕾丝边连衣裙";
//        labels[5] = "一片生机勃勃的动漫山地";
//        labels[6] = "水面上矗立着参天大树，天空壮阔无垠，金色的草地细腻逼真";
//        labels[7] = "一个身材火辣、胸部丰满、细节精致的美女。";
//        labels[8] = "一只狗";
//        labels[9] = "一只可爱的柯基犬正在海底散步";
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
        loadLabel_offset(tokenizer, label, mask, 0, labels[0], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 1, labels[1], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 2, labels[2], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 3, labels[3], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 4, labels[4], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 5, labels[5], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 6, labels[6], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 7, labels[7], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 8, labels[8], maxContextLen);
        loadLabel_offset(tokenizer, label, mask, 9, labels[9], maxContextLen);
        label.hostToDevice();
        mask.hostToDevice();
        condInput = t5.forward(label, mask);
        
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
//        		labels[0] = "母女俩在室内共享温馨时刻，都穿着复古风格的亚麻连衣裙，带有花卉图案和柔和的淡色调。母亲的裙子上搭配了一条精致的项链，女儿的头发上系着一个粉红色的蝴蝶结。";
//                labels[1] = "狐狸正在一个巨大的透明灯泡里睡觉";
//                labels[2] = "美丽的女孩，头发如瀑布般倾泻而下";
//                labels[3] = "青花瓷女孩的脸庞破碎，质地细腻。超现实";
//                labels[4] = "游戏艺术——一个拥有不同地理特性的岛屿，以及多个漂浮在太空中的小城市";
//                labels[5] = "一个由珊瑚礁构成的、渲染得绚丽的纸艺世界，其中充满了五彩斑斓的鱼类和海洋生物。";
//                labels[6] = "金发的美丽女孩，既酷炫又阳光";
//                labels[7] = "可爱的狐狸坐在金黄色的秋叶中，蓬松的尾巴蜷缩在身边，周围环绕着鲜艳的橙色花朵，阳光透过树木斑驳地洒落下来。";
//                labels[8] = "赛博朋克风格的熊猫正在街上散步";
//                labels[9] = "快乐的梦幻猫头鹰怪物坐在树枝上，周围是五彩斑斓的闪亮颗粒，背景是森林，羽毛细节精致.";
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
                loadLabel_offset(tokenizer, label, mask, 0, labels[0], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 1, labels[1], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 2, labels[2], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 3, labels[3], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 4, labels[4], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 5, labels[5], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 6, labels[6], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 7, labels[7], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 8, labels[8], maxContextLen);
                loadLabel_offset(tokenizer, label, mask, 9, labels[9], maxContextLen);
                label.hostToDevice();
                mask.hostToDevice();
                condInput = t5.forward(label, mask);
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
	
    public static void loadLabel_offset(BPETokenizerEN tokenizer, Tensor label, int index, String labelStr, int maxContextLen) {
    	int[] ids = tokenizer.encodeInt(labelStr, maxContextLen);
        for (int j = 0; j < maxContextLen; j++) {
            if (j < ids.length) {
                label.data[index * maxContextLen + j] = ids[j];
            } else {
                label.data[index * maxContextLen + j] = 0;
            }
        }
    }
	
	public static void test_omega_full_path_drop_cfg() throws Exception {

        int imgSize = 256;
        int t5MaxContextLen = 120;
        int clipMaxContextLen = 77;
        int batchSize = 10;
        
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        
        String tokenizer_path = "D:\\models\\t5\\spiece.model";
		SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path);
		int voc_size = 250112;
		int num_layers = 24;
		int head_num = 32;
		int t5EmbedDim = 2048;
		int d_ff = 5120;
		T5Encoder t5 = new T5Encoder(LossType.MSE, UpdaterType.adamw, voc_size, num_layers, head_num, t5MaxContextLen, t5EmbedDim, d_ff, false);
		t5.CUDNN = true;
		t5.RUN_MODEL = RunModel.EVAL;
		String t5_path = "D://models//t5//t5_encoder.model";
        ModelUtils.loadModel(t5, t5_path);

		String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
	    String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
	    BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int clipEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, clipMaxContextLen, vocabSize, clipEmbedDim, clipMaxContextLen, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = clipMaxContextLen;
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
        
        OmegaDiTFullLabel network = new OmegaDiTFullLabel(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, clipEmbedDim, clipMaxContextLen, t5EmbedDim, t5MaxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D://models//dit_txt//flux_sprint_b1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor clipLabel = new Tensor(batchSize * clipMaxContextLen, 1, 1, 1, true);
        Tensor t5Label = new Tensor(batchSize * t5MaxContextLen, 1, 1, 1, true);
       
        Tensor clipCondInput = null;
        Tensor clipCondInput_ynull = null;
        Tensor t5CondInput = null;
        Tensor t5CondInput_ynull = null;
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

//        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.68802416f,0.23106125f,0.34798032f,-0.11941326f,-0.964228f,-1.2219781f,-0.13273178f,0.39500353f,1.1611824f,0.33286065f,-0.9072468f,1.1964253f,0.77751267f,1.4342179f,-0.27347723f,0.6652678f,-1.0584584f,0.39614016f,0.10801358f,-1.3052012f,0.13688391f,-0.15965684f,-0.6702366f,0.68603706f,2.3822904f,1.4025455f,-0.81627595f,-0.32144645f,-0.109326765f,-1.3960866f,-0.38953456f,0.46669957f}, true);
//        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.117937f,4.2716165f,3.3511283f,3.7278311f,3.8668838f,3.888655f,2.970441f,3.5744536f,3.8475676f,3.9073176f,3.7287266f,3.8889313f,3.85656f,3.6298664f,3.286395f,3.3439715f,4.0913005f,3.8466156f,3.4434404f,3.1861749f,3.2535207f,3.0563624f,3.8309762f,4.52857f,2.9302385f,3.4940221f,4.7427244f,3.41022f,3.4938436f,3.7268739f,4.6008964f,4.3314376f}, true);
        
        Tensor mask = new Tensor(batchSize, 1, 1, t5MaxContextLen, true);
        
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
        loadLabel_offset(tokenizer, t5Label, mask, 0, labels[0], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 1, labels[1], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 2, labels[2], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 3, labels[3], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 4, labels[4], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 5, labels[5], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 6, labels[6], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 7, labels[7], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 8, labels[8], t5MaxContextLen);
        loadLabel_offset(tokenizer, t5Label, mask, 9, labels[9], t5MaxContextLen);
        t5Label.hostToDevice();
        mask.hostToDevice();
        t5CondInput = t5.forward(t5Label, mask);
        
        loadLabel_offset(bpe, clipLabel, 0, labels[0], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 1, labels[1], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 2, labels[2], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 3, labels[3], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 4, labels[4], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 5, labels[5], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 6, labels[6], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 7, labels[7], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 8, labels[8], clipMaxContextLen);
        loadLabel_offset(bpe, clipLabel, 9, labels[9], clipMaxContextLen);
        clipLabel.hostToDevice();
        clipCondInput = clip.get_full_clip_prompt_embeds(clipLabel);
        
        if(t5CondInput_ynull == null) {
        	t5CondInput_ynull = Tensor.createGPUTensor(t5CondInput_ynull, t5CondInput.number, t5CondInput.channel, t5CondInput.height, t5CondInput.width, true);
            Tensor y_null = network.main.t5LabelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null, t5CondInput_ynull, part_input_size, 0, 1, 0, 1);
            }
        	clipCondInput_ynull = Tensor.createGPUTensor(clipCondInput_ynull, clipCondInput.number, clipCondInput.channel, clipCondInput.height, clipCondInput.width, true);
            Tensor y_null2 = network.main.clipLabelEmbd.getY_embedding();
            int part_input_size2 = y_null2.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	network.tensorOP.op.copy_gpu(y_null2, clipCondInput_ynull, part_input_size2, 0, 1, 0, 1);
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
        		loadLabel_offset(tokenizer, t5Label, mask, 0, labels[0], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 1, labels[1], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 2, labels[2], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 3, labels[3], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 4, labels[4], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 5, labels[5], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 6, labels[6], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 7, labels[7], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 8, labels[8], t5MaxContextLen);
                loadLabel_offset(tokenizer, t5Label, mask, 9, labels[9], t5MaxContextLen);
                t5Label.hostToDevice();
                mask.hostToDevice();
                t5CondInput = t5.forward(t5Label, mask);
                
                loadLabel_offset(bpe, clipLabel, 0, labels[0], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 1, labels[1], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 2, labels[2], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 3, labels[3], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 4, labels[4], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 5, labels[5], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 6, labels[6], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 7, labels[7], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 8, labels[8], clipMaxContextLen);
                loadLabel_offset(bpe, clipLabel, 9, labels[9], clipMaxContextLen);
                clipLabel.hostToDevice();
                clipCondInput = clip.get_full_clip_prompt_embeds(clipLabel);
        	}
        	
        	System.out.println("start create test images.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, clipCondInput, clipCondInput_ynull, t5CondInput, t5CondInput_ynull, cos, sin, latend, eps, 1.0f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t,clipCondInput, clipCondInput_ynull, t5CondInput, t5CondInput_ynull, cos, sin, latend, eps, 2.5f);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\flux_path_drop_sprint-2\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_omega_sprint_path_drop_cfg_fluxvae() throws Exception {
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

        int latendDim = 16;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE vae = new Flux_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\e2e-flux-vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt_flux\\flux_sprint_b1_20.model";
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

        Tensor latend_mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.0355735719203949f,0.0048736147582530975f,-0.003450946183875203f,-0.03965449333190918f,0.036058880388736725f,0.04668542370200157f,-0.05957012623548508f,-0.001957265892997384f,-0.025739729404449463f,0.008893698453903198f,0.016873272135853767f,-0.0028637342620640993f,0.020259900018572807f,-0.005238725338131189f,-0.024182798340916634f,-0.0040397001430392265f}, true);
        Tensor latend_std = new Tensor(latendDim, 1, 1, 1, new float[] {2.339289665222168f,2.355745315551758f,2.364927053451538f,2.34753680229187f,2.3591816425323486f,2.38031268119812f,2.348323345184326f,2.343754291534424f,2.363278865814209f,2.3746819496154785f,2.3765056133270264f,2.362192392349243f,2.3591220378875732f,2.3649189472198486f,2.345620632171631f,2.3334505558013916f}, true);

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
            
            network.tensorOP.sub(sample, 0.1159f, sample);
            network.tensorOP.div(sample, 0.3611f, sample);
            icplan.latend_un_norm(sample, latend_mean, latend_std);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i, result, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.5f);
            
            network.tensorOP.sub(sample, 0.1159f, sample);
            network.tensorOP.div(sample, 0.3611f, sample);
            icplan.latend_un_norm(sample, latend_mean, latend_std);

            result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_fluxvae\\omega_path_drop_sprint\\" + i + "_T", result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_omega_sprint_path_drop_cfg_fluxvae_512() throws Exception {
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

        int latendDim = 16;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        Flux_VAE vae = new Flux_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\e2e-flux-vae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 64;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaDiT network = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, 768, token_drop, path_drop_prob, y_prob);
        network.CUDNN = true;
        network.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt_flux_512\\flux_sprint_b1_2.model";
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

        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.0355735719203949f,0.0048736147582530975f,-0.003450946183875203f,-0.03965449333190918f,0.036058880388736725f,0.04668542370200157f,-0.05957012623548508f,-0.001957265892997384f,-0.025739729404449463f,0.008893698453903198f,0.016873272135853767f,-0.0028637342620640993f,0.020259900018572807f,-0.005238725338131189f,-0.024182798340916634f,-0.0040397001430392265f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {2.339289665222168f,2.355745315551758f,2.364927053451538f,2.34753680229187f,2.3591816425323486f,2.38031268119812f,2.348323345184326f,2.343754291534424f,2.363278865814209f,2.3746819496154785f,2.3765056133270264f,2.362192392349243f,2.3591220378875732f,2.3649189472198486f,2.345620632171631f,2.3334505558013916f}, true);

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
                
                network.tensorOP.sub(sample, 0.1159f, sample);
                network.tensorOP.div(sample, 0.3611f, sample);
                icplan.latend_un_norm(sample, latendMean, latendStd);

                Tensor result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_512\\"+i+"_" + it, result, mean, std);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);
                
                network.tensorOP.sub(sample, 0.1159f, sample);
                network.tensorOP.div(sample, 0.3611f, sample);
                icplan.latend_un_norm(sample, latendMean, latendStd);

                result = vae.decode(sample);
                
                JCuda.cudaDeviceSynchronize();
                
                result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

                showImgs("D:\\test\\dit_fluxvae\\flux_path_drop_sprint_512\\"+i+"_" + it + "_T", result, mean, std);
                
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
	
    public static void test_t5() throws IOException {
    	
    	String tokenizer_path = "D:\\models\\t5\\spiece.model";

		SentencePieceTokenizer t = new SentencePieceTokenizer(tokenizer_path);

		String txt = "一只猫在苹果树下游泳";

		String[] tokens = t.tokenize(txt);
		
		System.out.println(JsonUtils.toJson(tokens));

		int[] tokensInt = t.encodeInt(txt, 120);
		
		System.out.println(JsonUtils.toJson(tokensInt));

		int[] idx = t.encodeInt(txt);

		System.out.println(JsonUtils.toJson(idx));

		String outText = t.decode(idx);

		System.out.println(outText); 
		
		int time = 120;
		int voc_size = 250112;
		int num_layers = 24;
		int head_num = 32;
		int embed_size = 2048;
		int d_ff = 5120;
		T5Encoder t5 = new T5Encoder(LossType.MSE, UpdaterType.adamw, voc_size, num_layers, head_num, time, embed_size, d_ff, false);
		t5.CUDNN = true;
		t5.RUN_MODEL = RunModel.EVAL;
    	
		String model_path = "D://models//t5//t5_encoder.model";
        ModelUtils.loadModel(t5, model_path);
		
//		Map<String, Tensor> weights = new HashMap<String, Tensor>();
//		weights.put("encoder.embed_tokens.weight", t5.stack.embed_tokens.weight);
//		
//		for(int i=0;i<num_layers;i++) {
//			weights.put("encoder.block."+i+".layer.0.SelfAttention.q.weight", t5.stack.block.get(i).attn.getqLinerLayer().weight);
//			weights.put("encoder.block."+i+".layer.0.SelfAttention.k.weight", t5.stack.block.get(i).attn.getkLinerLayer().weight);
//			weights.put("encoder.block."+i+".layer.0.SelfAttention.v.weight", t5.stack.block.get(i).attn.getvLinerLayer().weight);
//			weights.put("encoder.block."+i+".layer.0.SelfAttention.o.weight", t5.stack.block.get(i).attn.getoLinerLayer().weight);
//			if(i == 0) {
//				weights.put("encoder.block."+i+".layer.0.SelfAttention.relative_attention_bias.weight", t5.stack.block.get(i).relative_attention_bias.weight);
//			}
//			weights.put("encoder.block."+i+".layer.0.layer_norm.weight", t5.stack.block.get(i).norm.gamma);
//			
//			weights.put("encoder.block."+i+".layer.1.DenseReluDense.wi_0.weight", t5.stack.block.get(i).ffn.dr.linear1.weight);
//			weights.put("encoder.block."+i+".layer.1.DenseReluDense.wi_1.weight", t5.stack.block.get(i).ffn.dr.linear2.weight);
//			weights.put("encoder.block."+i+".layer.1.DenseReluDense.wo.weight", t5.stack.block.get(i).ffn.dr.linear3.weight);
//			weights.put("encoder.block."+i+".layer.1.layer_norm.weight", t5.stack.block.get(i).ffn.norm.gamma);
//		}
//		weights.put("encoder.final_layer_norm.weight", t5.stack.final_layer_norm.gamma);
//		
//        String t5Weight = "D:\\models\\t5\\t5_encoder.json";
//        List<String> igones = new ArrayList<String>();
//        igones.add("shared.weight");
//        LagJsonReader.readJsonFileBigWeightIterator(t5Weight, weights, igones);
//        
//        for(int i=0;i<num_layers;i++) {
//        	t5.stack.block.get(i).norm.gamma = weights.get("encoder.block."+i+".layer.0.layer_norm.weight");
//        	t5.stack.block.get(i).ffn.norm.gamma = weights.get("encoder.block."+i+".layer.1.layer_norm.weight");
//        }
//        t5.stack.final_layer_norm.gamma = weights.get("encoder.final_layer_norm.weight");
//		
//        System.err.println(t5.stack.final_layer_norm.gamma);
        
        Tensor input = new Tensor(time, 1, 1, 1, true);
        Tensor mask = new Tensor(1, 1, 1, time, true);
        float max = -3.4028e+38f;
        for(int i = 0;i<time;i++) {
        	input.data[i] = tokensInt[i];
        	if(input.data[i] == 0) {
        		mask.data[i] = max;
        	}
        }
        input.hostToDevice();
        mask.hostToDevice();
        Tensor output = t5.forward(input, mask);
        output.showDM("output");
        

//        String save_model_path = "D://models//t5//t5_encoder.model";
//        ModelUtils.saveModel(t5, save_model_path);
        
        
	}
    
    public static void omega_full_b1_iddpm_test() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
		String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
		String t5DataPath = "D:\\dataset\\amine\\vavae_t5.bin";
        
        int batchSize = 4;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int clip_len = 77;
        int t5_len = 120;
        int clipEmbedDim = 768;
        int t5EmbedDim = 2048;
        
        LatendDataset_t5_clip dataLoader = new LatendDataset_t5_clip(dataPath, clipDataPath, t5DataPath, batchSize, latendDim, height, width, clip_len, clipEmbedDim, t5_len, t5EmbedDim, BinDataType.float32);

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

        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 1, 2, 2, 4};
        int ch = 128;
        VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, 256, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\vavae.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        Tensor latend = new Tensor(batchSize, dataLoader.channel, dataLoader.height, dataLoader.width, true);

        Tensor clipInput = new Tensor(batchSize * dataLoader.clipMaxTime, 1, 1, dataLoader.clipEmbd, true);
        Tensor t5Input = new Tensor(batchSize * dataLoader.t5MaxTime, 1, 1, dataLoader.t5Embd, true);
        
        Tensor img = new Tensor(batchSize, 3, dataLoader2.img_h, dataLoader2.img_w, true);
        Tensor de_img = new Tensor(batchSize, 3, dataLoader2.img_h, dataLoader2.img_w, true);
        
        int[][] indexs = dataLoader.order();
        
        int[] next = indexs[0];
        int it = 120;
        if(it < indexs.length - 1) {
        	next = indexs[it+1];
        }
        dataLoader.loadData(indexs[it], next, latend, clipInput, t5Input, it);
        dataLoader2.loadData(indexs[it], next, img, it);
        vae.tensorOP.copyGPU(img, de_img);
        de_img.syncHost();
        showImgs("D:\\test\\dit_vavae\\doin_test\\", de_img, mean, std);
        
        Tensor result = vae.decode(latend);
        
        JCuda.cudaDeviceSynchronize();
        
        result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
        
        float[] vae_mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] vae_std = new float[]{0.5f, 0.5f, 0.5f};
        
        showImgs("D:\\test\\dit_vavae\\flux_path_test\\", result, vae_mean, vae_std);

        next = indexs[0];
        it++;
        if(it < indexs.length - 1) {
        	next = indexs[it+1];
        }
        dataLoader.loadData(indexs[it], next, latend, clipInput, t5Input, it);
        dataLoader2.loadData(indexs[it], next, img, it);
        
        vae.tensorOP.copyGPU(img, de_img);
        
        result = vae.decode(latend);
        
        JCuda.cudaDeviceSynchronize();
        
        result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
        
        showImgs("D:\\test\\dit_vavae\\flux_path_test\\t", result, vae_mean, vae_std);
        de_img.syncHost();
        showImgs("D:\\test\\dit_vavae\\doin_test\\t", de_img, mean, std);
    }
    
    public static void main(String[] args) {
		 
	        try {
	           
	        	omega_sprint_b1_iddpm_train();  // 256 train
	        	
//	        	omega_sprint_b1_iddpm_train_512(); // 512 fine turn
	        	
//	        	test_omega_sprint_path_drop_cfg(); // simple 256
	        	
//	        	test_omega_sprint_path_drop_cfg_512(); // simple 512
	        	
//	        	omega_sprint_b1_iddpm_train_fluxvae();
	        	
//	        	test_omega_sprint_path_drop_cfg_fluxvae();
	        	
//	        	omega_sprint_b1_iddpm_train_fluxvae_512();
	        	
//	        	test_omega_sprint_path_drop_cfg_fluxvae_512();
	        	
//	        	test_t5();
	        	
//	        	test_omega_t5_path_drop_cfg();
	        	
//	        	omega_full_b1_iddpm_train();
	        	
//	        	test_omega_full_path_drop_cfg();
	        	
//	        	omega_full_b1_iddpm_test();
	        	
	        } catch (Exception e) {
	            // TODO: handle exception
	            e.printStackTrace();
	        } finally {
	            // TODO: handle finally clause
	            CUDAMemoryManager.free();
	        }
	  }
	
}
