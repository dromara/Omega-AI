package com.omega.example.dit.test;

import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.JiT;
import com.omega.engine.nn.network.dit.JiT_Sprint;
import com.omega.engine.nn.network.dit.JiT_Sprint2;
import com.omega.engine.nn.network.dit.MMJiT;
import com.omega.engine.nn.network.dit.MMJiT_REPA;
import com.omega.engine.nn.network.dit.MMJiT_Sprint;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.ImageClipDataLoader;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.sd.utils.SDImageLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.runtime.JCuda;

public class MMJiTTest {

	public static void mmjit_b16_iddpm_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 18;
//        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
//        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        int headNum = 12;
        int txt_depth = 2;
        int depth = 12;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        MMJiT jit = new MMJiT(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, headNum, txt_depth, depth, textEmbedDim, maxContextLen, y_prob);
        jit.CUDNN = true;
        jit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 20, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_MMJiT_ICPlan(dataLoader, icplan, "D://models//jit//", 1);
        String save_model_path = "D://models//jit//jit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void mmjit_b16_repa_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 24;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

		String repaImgDirPath = "D:\\dataset\\images_224_224\\";

        int repaImgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, repaImgDirPath, ".jpg", repaImgSize, repaImgSize, batchSize, false, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, repaImgSize, repaImgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
        
        int ditHeadNum = 12;
        int txt_depth = 2;
        int depth = 12;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        MMJiT_REPA jit = new MMJiT_REPA(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, ditHeadNum, dinov_hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob);
        jit.CUDNN = true;
        jit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 20, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_MMJiT_ICPlan(dinov, dataLoader, dataLoader2, icplan, "D://models//jit//", 1);
        String save_model_path = "D://models//jit//mmjit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void mmjit_b16_sprint_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 24;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

		String repaImgDirPath = "D:\\dataset\\images_224_224\\";

        int repaImgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, repaImgDirPath, ".jpg", repaImgSize, repaImgSize, batchSize, false, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, repaImgSize, repaImgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
        
        int ditHeadNum = 12;
        int txt_depth = 2;
        int depth = 12;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        MMJiT_Sprint jit = new MMJiT_Sprint(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, ditHeadNum, dinov_hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        jit.CUDNN = true;
        jit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 20, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_MMJiT_Sprint(dinov, dataLoader, dataLoader2, icplan, "D://models//jit//", 1);
        String save_model_path = "D://models//jit//mmjit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void jit_b16_sprint_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = false;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 24;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

		String repaImgDirPath = "D:\\dataset\\images_224_224\\";

        int repaImgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, repaImgDirPath, ".jpg", repaImgSize, repaImgSize, batchSize, false, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, repaImgSize, repaImgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
        
        int ditHeadNum = 12;
        int txt_depth = 2;
        int depth = 12;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        JiT_Sprint jit = new JiT_Sprint(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, ditHeadNum, dinov_hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        jit.CUDNN = true;
        jit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 20, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_JiT_Sprint(dinov, dataLoader, dataLoader2, icplan, "D://models//jit//", 1);
        String save_model_path = "D://models//jit//jit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void jit_b16_sprint2_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = false;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 28;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

		String repaImgDirPath = "D:\\dataset\\images_224_224\\";

        int repaImgSize = 224;

        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        SDImageLoader dataLoader2 = new SDImageLoader(labelPath, repaImgDirPath, ".jpg", repaImgSize, repaImgSize, batchSize, false, mean, std);
		
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, repaImgSize, repaImgSize, dinov_patchSize, dinov_hiddenSize, headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
        
        int ditHeadNum = 12;
        int depth = 12;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        JiT_Sprint2 jit = new JiT_Sprint2(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, ditHeadNum, dinov_hiddenSize, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        jit.CUDNN = true;
        jit.learnRate = 1e-4f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 20, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_JiT_Sprint2(dinov, dataLoader, dataLoader2, icplan, "D://models//jit//", 1);
        String save_model_path = "D://models//jit//jit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void loadWeight(Map<String, Object> weightMap, JiT dit, int depth, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
    }
	
	public static void test_mmjit_b16_cfg() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        
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

        int jitHeadNum = 12;
        int txt_depth = 2;
        int depth = 17;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        MMJiT network = new MMJiT(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, txt_depth, depth, textEmbedDim, maxContextLen, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 100, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        int theta = 10000;

        Tensor[] cs1d = RoPEKernel.create1DRope(network.maxContextLen, network.headDims, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];

        Tensor[] cs2d = RoPEKernel.create2DRope(network.headNum, network.time, network.headDims, network.grid, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];

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
                for(int b = 0;b<batchSize * maxContextLen;b++) {
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
                }
            }

            for(int it = 0;it<5;it++) {
            	
            	System.out.println("start create test images.");

                GPUOP.getInstance().cudaRandn(noise);
//                network.tensorOP.mul(noise, 2.0f, noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, eps, 1.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it, sample);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_cfg(network, noise2, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, eps, 2.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it + "_T", sample);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void test_mmjit_repa_b16_cfg() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        
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

        int jitHeadNum = 12;
        int txt_depth = 2;
        int depth = 17;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        MMJiT_REPA network = new MMJiT_REPA(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 100, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        int theta = 10000;

        Tensor[] cs1d = RoPEKernel.create1DRope(network.maxContextLen, network.headDims, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];

        Tensor[] cs2d = RoPEKernel.create2DRope(network.headNum, network.time, network.headDims, network.grid, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];

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
                for(int b = 0;b<batchSize * maxContextLen;b++) {
                	network.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
                }
            }

            for(int it = 0;it<5;it++) {
            	
            	System.out.println("start create test images.");

                GPUOP.getInstance().cudaRandn(noise);
//                network.tensorOP.mul(noise, 2.0f, noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_cfg(network, noise, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, eps, 1.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it, sample);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_cfg(network, noise2, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, eps, 2.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it + "_T", sample);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void test_mmjit_sprint_b16_cfg() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;

        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        
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

        int jitHeadNum = 12;
        int txt_depth = 2;
        int depth = 17;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        MMJiT_Sprint network = new MMJiT_Sprint(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 100, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        int theta = 10000;

        Tensor[] cs1d = RoPEKernel.create1DRope(network.maxContextLen, network.headDims, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];

        Tensor[] cs2d = RoPEKernel.create2DRope(network.headNum, network.time, network.headDims, network.grid, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];

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
//                network.tensorOP.mul(noise, 2.0f, noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, latend, eps, 1.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it, sample);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, latend, eps, 2.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it + "_T", sample);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void test_jit_sprint_b16_cfg() throws Exception {
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;

        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        
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

        int jitHeadNum = 12;
        int txt_depth = 2;
        int depth = 17;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        JiT_Sprint network = new JiT_Sprint(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, hiddenSize, txt_depth, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 100, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_1.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor eps = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        int theta = 10000;

        Tensor[] cs1d = RoPEKernel.create1DRope(network.maxContextLen, network.headDims, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];

        Tensor[] cs2d = RoPEKernel.create2DRope(network.headNum, network.time, network.headDims, network.grid, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];

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
//                network.tensorOP.mul(noise, 2.0f, noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, latend, eps, 1.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it, sample);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos1d, sin1d, cos2d, sin2d, latend, eps, 2.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it + "_T", sample);
                
                System.out.println("finish create.");
            }
        	
        }
        
	}
	
	public static void test_jit_sprint2_b16_cfg() throws Exception {
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;

        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        
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

        int jitHeadNum = 12;
        int depth = 17;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        float path_drop_prob = 0.05f;
        
        JiT_Sprint2 network = new JiT_Sprint2(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, hiddenSize, depth, textEmbedDim, maxContextLen, y_prob, path_drop_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 100, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_1.model";
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
//                network.tensorOP.mul(noise, 2.0f, noise);
                noise.copyGPU(noise2);
                
                Tensor sample = icplan.forward_with_path_drop_cfg(network, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it, sample);
                
                System.out.println("finish create.");
                
                sample = icplan.forward_with_path_drop_cfg(network, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);
                JCuda.cudaDeviceSynchronize();
                
                sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

                showImgs("D:\\test\\mmjit\\256\\"+i+"_" + it + "_T", sample);
                
                System.out.println("finish create.");
            }
        	
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
	
    public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb3(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }
    
    public static void showImgs(String outputPath, Tensor input) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb(once, input.channel, input.height, input.width), input.height, input.width, null, null);
        }
    }
    
    public static void test_img() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 4;
        float[] mean = new float[]{0.0f, 0.0f, 0.0f};
        float[] std = new float[]{1f, 1f, 1f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, mean, std);
    
        Tensor imgs = new Tensor(batchSize, channel, imgSize, imgSize, true);
        
        Tensor condInput = new Tensor(batchSize * maxContextLen, 1, 1, textEmbedDim, true);
        
        int[][] indexs = dataLoader.shuffle();
        
        for(int it = 0;it<10;it++) {
        	
        	int[] next = indexs[0];
            if(it < indexs.length - 1) {
            	next = indexs[it+1];
            }
        	
        	dataLoader.loadData(indexs[it], next, imgs, condInput, it);
        	
        	imgs.data = MatrixOperation.clampSelf(imgs.syncHost(), -1, 1);

            showImgs("D:\\test\\mmjit\\256\\"+it+"_" + it, imgs, mean, std);
        	
        }

    }
	
	public static void main(String[] args) {
		 
		try {
		   
//			test_img();
			
//			mmjit_b16_iddpm_train();
			
//			jit_b16_sprint_train();
			
			jit_b16_sprint2_train();
			
//			test_jit_sprint2_b16_cfg();
			
//			test_mmjit_b16_cfg();

			
		} catch (Exception e) {
		    // TODO: handle exception
		    e.printStackTrace();
		} finally {
		    // TODO: handle finally clause
		    CUDAMemoryManager.free();
		}
	}
	
}
