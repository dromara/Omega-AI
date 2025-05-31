package com.omega.example.dit.test;

import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.DiTBlock;
import com.omega.engine.nn.layer.dit.modules.DiTCrossAttentionLayer2;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.DiT;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

public class DiTTest {
	
	public static void loadWeight(Map<String, Object> weightMap, DiT network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ClipModelUtils.loadData(network.main.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
        ClipModelUtils.loadData(network.main.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ClipModelUtils.loadData(network.main.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ClipModelUtils.loadData(network.main.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ClipModelUtils.loadData(network.main.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ClipModelUtils.loadData(network.main.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        for(int i = 0;i<6;i++){
        	DiTBlock block = network.main.blocks.get(i);
        	block.norm1.gamma = ClipModelUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight");
        	ClipModelUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn1.qL.weight");
            ClipModelUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn1.qL.bias");
        	ClipModelUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn1.kL.weight");
            ClipModelUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn1.kL.bias");
        	ClipModelUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn1.vL.weight");
            ClipModelUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn1.vL.bias");
        	ClipModelUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn1.proj.weight");
            ClipModelUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn1.proj.bias");
        	block.norm2.gamma = ClipModelUtils.loadData(block.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight");
        	ClipModelUtils.loadData(block.modulation.weight, weightMap, "blocks."+i+".default_modulation.1.weight");
            ClipModelUtils.loadData(block.modulation.bias, weightMap, "blocks."+i+".default_modulation.1.bias");
            ClipModelUtils.loadData(block.cross_attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn2.query.weight");
            ClipModelUtils.loadData(block.cross_attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn2.query.bias");
        	ClipModelUtils.loadData(block.cross_attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn2.key.weight");
            ClipModelUtils.loadData(block.cross_attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn2.key.bias");
        	ClipModelUtils.loadData(block.cross_attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn2.value.weight");
            ClipModelUtils.loadData(block.cross_attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn2.value.bias");
        	ClipModelUtils.loadData(block.cross_attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn2.out_proj.weight");
            ClipModelUtils.loadData(block.cross_attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn2.out_proj.bias");
            block.norm3.gamma = ClipModelUtils.loadData(block.norm3.gamma, weightMap, 1, "blocks."+i+".norm3.weight");
            ClipModelUtils.loadData(block.mlp.linear1.weight, weightMap, "blocks."+i+".mlp.fc1.weight");
            ClipModelUtils.loadData(block.mlp.linear1.bias, weightMap, "blocks."+i+".mlp.fc1.bias");
        	ClipModelUtils.loadData(block.mlp.linear2.weight, weightMap, "blocks."+i+".mlp.fc2.weight");
            ClipModelUtils.loadData(block.mlp.linear2.bias, weightMap, "blocks."+i+".mlp.fc2.bias");
        }
        
        network.main.finalLayer.finalNorm.gamma = ClipModelUtils.loadData(network.main.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight");
        
        ClipModelUtils.loadData(network.main.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ClipModelUtils.loadData(network.main.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        
        ClipModelUtils.loadData(network.main.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation1.weight");
        ClipModelUtils.loadData(network.main.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation1.bias");
        ClipModelUtils.loadData(network.main.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation2.weight");
        ClipModelUtils.loadData(network.main.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation2.bias");

    }
	
	public static void dit_test() {
		
		int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 6;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, 4, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 77, 512, mlpRatio);
		
        int batchSize = 2;
        int channel = 4;
        int z_dims = 32;
        Tensor x = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.order(batchSize * channel * z_dims * z_dims, 0.01f, 0.01f), true);

        Tensor tx = new Tensor(batchSize, 1, 1, 1, new float[] {0, 2}, true);
        
        int cTime = 77;
        int cDims = 512;
        Tensor cx = new Tensor(batchSize * cTime, 1, 1, cDims, MatrixUtils.order(batchSize * cTime * cDims, 0.01f, 0.01f), true);
        
        System.out.println(dit.hiddenSize);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(dit.time, dit.hiddenSize, dit.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        cos.showDM("cos");
        sin.showDM("sin");
        
        String weight = "H:\\model\\dit.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), dit, true);
        
        dit.forward(x, tx, cx, cos, sin);
        
        dit.getOutput().showDM();
        dit.getOutput().showShape();

        Tensor dy = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.order(batchSize * channel * z_dims * z_dims, 0.01f, 0.01f), true);
        
        dit.back(dy, cos, sin);
        
	}
	
	public static void dit_train() throws Exception {
        String labelPath = "I:\\dataset\\sd-anime\\anime_op\\data.json";
        String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 2;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
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
        String clipWeight = "H:\\model\\clip-vit-base-patch32.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
        int z_dims = 128;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 2, 4};
        int ch = 128;
        VQVAE2 vae = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 4;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_Anime2(dataLoader, vae, clip);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_pokemon_train() throws Exception {
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
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
        String clipWeight = "H:\\model\\clip-vit-base-patch32.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
        int z_dims = 32;
        int latendDim = 16;
        int num_vq_embeddings = 512;
        int num_res_blocks = 1;
        int[] ch_mult = new int[]{1, 2, 2, 4};
        int ch = 32;
        VQVAE2 vae = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\pokemon_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 6;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_Anime(dataLoader, vae, clip);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	 public static void main(String[] args) {
	        try {
	           
//	        	dit_test();
	        	
//	        	dit_train();
	        	
	        	dit_pokemon_train();
	        	
	        } catch (Exception e) {
	            // TODO: handle exception
	            e.printStackTrace();
	        } finally {
	            // TODO: handle finally clause
	            CUDAMemoryManager.free();
	        }
	    }
	
}
