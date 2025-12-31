package com.omega.example.dit.test;

import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.flux.FluxDiTBlock;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.JiT;
import com.omega.engine.nn.network.dit.JiT_REPA;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.ImageClipDataLoader;
import com.omega.example.dit.dataset.ImageClip_REAP_DataLoader;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.runtime.JCuda;

public class JiTTest {

	public static void jit_b16_iddpm_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 32;
//        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
//        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        int headNum = 12;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        JiT jit = new JiT(LossType.MSE_right, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, headNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, false, y_prob);
        jit.CUDNN = true;
        jit.learnRate = 0.00005f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
       
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_JiT_ICPlan(dataLoader, icplan, "D://models//jit//", 4);
        String save_model_path = "D://models//jit//jit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void loadWeight(Map<String, Object> weightMap, JiT dit, int depth, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.proj1.weight, weightMap, "x_embedder.proj1.weight");
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.proj2.weight, weightMap, "x_embedder.proj2.weight", 4);
        ModeLoaderlUtils.loadData(dit.main.patchEmbd.proj2.bias, weightMap, "x_embedder.proj2.bias");
        
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(dit.main.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");

        ModeLoaderlUtils.loadData(dit.main.labelEmbd.init_y_embedding(), weightMap, "y_embedder.y_embedding");
        
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
        ModeLoaderlUtils.loadData(dit.main.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");

        for(int i = 0;i<depth;i++) {
        	
        	FluxDiTBlock block = dit.main.blocks.get(i);
        	
        	block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
        	block.norm3.gamma = ModeLoaderlUtils.loadData(block.norm3.gamma, weightMap, 1, "blocks."+i+".norm2.weight");  
        	
        	block.attn.qNorm.gamma = ModeLoaderlUtils.loadData(block.attn.qNorm.gamma, weightMap, 1, "blocks."+i+".attn.q_norm.weight"); 
        	block.attn.kNorm.gamma = ModeLoaderlUtils.loadData(block.attn.kNorm.gamma, weightMap, 1, "blocks."+i+".attn.k_norm.weight"); 
        	ModeLoaderlUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn.q.weight");
            ModeLoaderlUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn.q.bias");
        	ModeLoaderlUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn.k.weight");
            ModeLoaderlUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn.k.bias");
        	ModeLoaderlUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn.v.weight");
            ModeLoaderlUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn.v.bias");
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
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_linear1.weight");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_linear1.bias");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_linear2.weight");
        ModeLoaderlUtils.loadData(dit.main.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_linear2.bias");
    }
	
	public static void test_jit_b16() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 2;
//        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
//        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, null, null);

        int headNum = 12;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        JiT jit = new JiT(LossType.MSE_right, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, headNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, false, y_prob);
        jit.CUDNN = true;
        jit.learnRate = 0.00005f;
        
        String weight = "D:\\models\\jit_b16.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jit, depth, true);
        
        Tensor noise = new Tensor(batchSize, channel, imgSize, imgSize, true);
        String zPath = "D:\\models\\jit_z.json";
 	    Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(zPath);
 	    ModeLoaderlUtils.loadData(noise, zdatas, "z");
 	    
 	    Tensor cy = new Tensor(batchSize * maxContextLen, 1, 1,textEmbedDim, true);
        String cyPath = "D:\\models\\jit_cy.json";
 	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
 	    ModeLoaderlUtils.loadData(cy, cydatas, "cy", 2);

        Tensor t = new Tensor(batchSize, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
 	    
        Tensor[] cs = RoPEKernel.getCosAndSin2D(jit.time, jit.hiddenSize, jit.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

 	    Tensor out = jit.forward(noise, t, cy, cos, sin);
 	    
 	    out.showDM("out");
 	    
 	    jit.back(noise, cos, sin);
 	    
//        ICPlan icplan = new ICPlan(jit.tensorOP);

//        String model_path = "D://models//jit//jit_b16_20.model";
//        ModelUtils.loadModel(jit, model_path);
       
//        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
//        
//        optimizer.train_JiT_ICPlan(dataLoader, icplan, "D://models//jit//", 4);
//        String save_model_path = "D://models//jit//jit_b16.model";
//        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void jit_repa_b16_iddpm_train() throws Exception {
		
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
		String imgDirPath_224 = "D:\\dataset\\images_224_224\\";
        boolean horizontalFilp = false;
        int imgSize = 256;
        int imgSize_224 = 224;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 24;
        float[] mean = new float[]{0f, 0f, 0f};
        float[] std = new float[]{1f, 1f, 1f};
        float[] mean2 = new float[]{0.485f, 0.456f, 0.406f};
        float[] std2 = new float[]{0.229f, 0.224f, 0.225f};
        ImageClip_REAP_DataLoader dataLoader = new ImageClip_REAP_DataLoader(labelPath, imgDirPath, imgDirPath_224, clipDataPath, ".jpg", imgSize, imgSize, imgSize_224, imgSize_224, maxContextLen, textEmbedDim, batchSize, horizontalFilp, mean, std, mean2, std2);
        
		int dinov_patchSize = 14;
		int dinov_hiddenSize = 768;
		int dinov_headNum = 12;
		int dinov_depth = 12;
		int dinov_mlpRatio = 4;
		Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize_224, imgSize_224, dinov_patchSize, dinov_hiddenSize, dinov_headNum, dinov_depth, dinov_mlpRatio);
		dinov.CUDNN = true;
		dinov.RUN_MODEL = RunModel.EVAL;
        
        String repa_model_path = "D:\\models\\dionv2-14-b.model";
        ModelUtils.loadModel(dinov, repa_model_path);
        
        int headNum = 12;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        int z_idx = 4;
        
        JiT_REPA jit = new JiT_REPA(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, headNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, z_idx, dinov_hiddenSize, false, y_prob);
        jit.CUDNN = true;
        jit.learnRate = 0.0001f;
        
        ICPlan icplan = new ICPlan(jit.tensorOP, 50, 0);

        //String model_path = "D://models//jit//jit_b16_4.model";
        //ModelUtils.loadModel(jit, model_path);
       
        MBSGDOptimizer optimizer = new MBSGDOptimizer(jit, 100, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        
        optimizer.train_JiT_REPA_ICPlan(dinov, dataLoader, icplan, "D://models//jit//", 4);
        String save_model_path = "D://models//jit//jit_b16.model";
        ModelUtils.saveModel(jit, save_model_path);
    }
	
	public static void test_jit_b16_cfg() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 10;
        float[] mean = new float[]{0.0f, 0.0f, 0.0f};
        float[] std = new float[]{1f, 1f, 1f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, mean, std);

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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        JiT network = new JiT(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_20.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;

        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat holding a sign that says hello world";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "the cambridge shoulder bag";
        labels[4] = "fruit cream cake";
        labels[5] = "a woman";
        labels[6] = "A panda sleep on the water";
        labels[7] = "a dog";
        labels[8] = "A yellow mushroom grows in the forest";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(bpe, label, 0, labels[0]);
        dataLoader.loadLabel_offset(bpe, label, 1, labels[1]);
        dataLoader.loadLabel_offset(bpe, label, 2, labels[2]);
        dataLoader.loadLabel_offset(bpe, label, 3, labels[3]);
        dataLoader.loadLabel_offset(bpe, label, 4, labels[4]);
        dataLoader.loadLabel_offset(bpe, label, 5, labels[5]);
        dataLoader.loadLabel_offset(bpe, label, 6, labels[6]);
        dataLoader.loadLabel_offset(bpe, label, 7, labels[7]);
        dataLoader.loadLabel_offset(bpe, label, 8, labels[8]);
        dataLoader.loadLabel_offset(bpe, label, 9, labels[9]);
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
            
            Tensor sample = icplan.forward_with_cfg(network, noise, condInput_ynull, cos, sin, 1.0f);

            network.tensorOP.add(sample, 1, sample);
            network.tensorOP.div(sample, 2, sample);

            JCuda.cudaDeviceSynchronize();
            
            sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);
            
            showImgs("D:\\test\\dit_vavae\\jit\\" + i, sample, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, condInput_ynull, cos, sin, 2.0f);

            network.tensorOP.add(sample, 1, sample);
            network.tensorOP.div(sample, 2, sample);

            JCuda.cudaDeviceSynchronize();
            
            sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\jit\\" + i + "_T", sample, mean, std);
            
            System.out.println("finish create.");

        }
        
	}
	
	public static void test_jit_repa_b16_cfg() throws Exception {
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_256_256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int channel = 3;
        int maxContextLen = 77;
        int textEmbedDim = 768;
        int batchSize = 10;
        float[] mean = new float[]{0.0f, 0.0f, 0.0f};
        float[] std = new float[]{1f, 1f, 1f};
        ImageClipDataLoader dataLoader = new ImageClipDataLoader(labelPath, imgDirPath, clipDataPath, ".jpg", imgSize, imgSize, maxContextLen, textEmbedDim, batchSize, horizontalFilp, mean, std);

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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 16;
        int bottleneck_dim = 128;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        int z_idx = 4;
		int dinov_hiddenSize = 768;
        
        JiT_REPA network = new JiT_REPA(LossType.MSE, UpdaterType.adamw, channel, imgSize, imgSize, patchSize, bottleneck_dim, hiddenSize, jitHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, z_idx, dinov_hiddenSize, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;

        ICPlan icplan = new ICPlan(network.tensorOP, 50, 0.0f);
        
        String model_path = "D://models//jit//jit_b16_20.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;

        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor noise2 = new Tensor(batchSize, network.inChannel, network.height, network.width, true);

        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        network.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "A cat holding a sign that says hello world";
        labels[1] = "a vibrant anime mountain lands";
        labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
        labels[3] = "the cambridge shoulder bag";
        labels[4] = "fruit cream cake";
        labels[5] = "a woman";
        labels[6] = "A panda sleep on the water";
        labels[7] = "a dog";
        labels[8] = "A yellow mushroom grows in the forest";
        labels[9] = "A lovely corgi is taking a walk under the sea";
        dataLoader.loadLabel_offset(bpe, label, 0, labels[0]);
        dataLoader.loadLabel_offset(bpe, label, 1, labels[1]);
        dataLoader.loadLabel_offset(bpe, label, 2, labels[2]);
        dataLoader.loadLabel_offset(bpe, label, 3, labels[3]);
        dataLoader.loadLabel_offset(bpe, label, 4, labels[4]);
        dataLoader.loadLabel_offset(bpe, label, 5, labels[5]);
        dataLoader.loadLabel_offset(bpe, label, 6, labels[6]);
        dataLoader.loadLabel_offset(bpe, label, 7, labels[7]);
        dataLoader.loadLabel_offset(bpe, label, 8, labels[8]);
        dataLoader.loadLabel_offset(bpe, label, 9, labels[9]);
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
            
            Tensor sample = icplan.forward_with_cfg(network, noise, condInput_ynull, cos, sin, 1.0f);

            network.tensorOP.add(sample, 1, sample);
            network.tensorOP.div(sample, 2, sample);

            JCuda.cudaDeviceSynchronize();
            
            sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);
            
            showImgs("D:\\test\\dit_vavae\\jit\\" + i, sample, mean, std);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_cfg(network, noise2, condInput_ynull, cos, sin, 10.0f);

            network.tensorOP.add(sample, 1, sample);
            network.tensorOP.div(sample, 2, sample);

            JCuda.cudaDeviceSynchronize();
            
            sample.data = MatrixOperation.clampSelf(sample.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\jit\\" + i + "_T", sample, mean, std);
            
            System.out.println("finish create.");

        }
        
	}
	
    public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb3(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }
    
	
	public static void main(String[] args) {
		 
		try {
		   
//			jit_b16_iddpm_train();
			
//			jit_repa_b16_iddpm_train();
			
//			test_jit_b16_cfg();
			
			test_jit_repa_b16_cfg();
			
//			test_jit_b16();
			
		} catch (Exception e) {
		    // TODO: handle exception
		    e.printStackTrace();
		} finally {
		    // TODO: handle finally clause
		    CUDAMemoryManager.free();
		}
	}
	
}
