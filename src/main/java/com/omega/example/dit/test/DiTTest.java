package com.omega.example.dit.test;

import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.DiTSkipBlock;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipText;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.DiT;
import com.omega.engine.nn.network.DiT_ORG;
import com.omega.engine.nn.network.DiT_SRA;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.sd.utils.SDImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.driver.JCudaDriver;

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
        
        ClipModelUtils.loadData(network.main.labelEmbd.linear1.weight, weightMap, "y_embedder.mlp.0.weight");
        ClipModelUtils.loadData(network.main.labelEmbd.linear1.bias, weightMap, "y_embedder.mlp.0.bias");
        ClipModelUtils.loadData(network.main.labelEmbd.linear2.weight, weightMap, "y_embedder.mlp.2.weight");
        ClipModelUtils.loadData(network.main.labelEmbd.linear2.bias, weightMap, "y_embedder.mlp.2.bias");
        
        for(int i = 0;i<6;i++){
        	DiTSkipBlock block = network.main.blocks.get(i);
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
            if(block.longSkip) {
            	block.skipNorm.gamma = ClipModelUtils.loadData(block.skipNorm.gamma, weightMap, 1, "blocks."+i+".skip_norm.weight");
            	ClipModelUtils.loadData(block.skipLinear.weight, weightMap, "blocks."+i+".skip_linear.weight");
                ClipModelUtils.loadData(block.skipLinear.bias, weightMap, "blocks."+i+".skip_linear.bias");
            }
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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT_SRA dit = new DiT_SRA(LossType.MSE, UpdaterType.adamw, 4, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 77, 512, mlpRatio, true, false, 4);
        dit.learnRate=1e-4f;
        int batchSize = 2;
        int channel = 4;
        int z_dims = 32;
        Tensor latend = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.order(batchSize * channel * z_dims * z_dims, -0.03f, 0.01f), true);

        Tensor tx = new Tensor(batchSize, 1, 1, 1, new float[] {0, 853}, true);
        
        int cTime = 77;
        int cDims = 512;
        Tensor cx = new Tensor(batchSize * cTime, 1, 1, cDims, MatrixUtils.order(batchSize * cTime * cDims, 0.01f, 0.01f), true);
        
        System.out.println(dit.hiddenSize);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(dit.time, dit.hiddenSize, dit.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        cos.showDM("cos");
        sin.showDM("sin");
        
        DiT_SRA teacher = new DiT_SRA(LossType.MSE, UpdaterType.adamw, 4, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 77, 512, mlpRatio, true, false, 8);
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        Tensor noise = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.val(batchSize * channel * z_dims * z_dims, 0.1f), true);
        
        Tensor x_t = new Tensor(batchSize, channel, z_dims, z_dims, true);
        
        iddpm.q_sample(latend, noise, x_t, tx);
        x_t.showDM();
        dit.forward(x_t, tx, cx, cos, sin);
        
        dit.getOutput().showDM();
        
        teacher.copyParams(dit);
        
        teacher.forward(x_t, tx, cx, cos, sin);
        
        teacher.getOutput().showDM();
        
//        String weight = "H:\\model\\dit3.json";
//        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), dit, true);
//        
////	    dit.forward(latend, tx, cx, cos, sin);
////	    dit.getOutput().showDM("now");
////	    dit.getOutput().showShape();
//        
//        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
//        
//        Tensor noise = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.val(batchSize * channel * z_dims * z_dims, 0.1f), true);
//        
//        Tensor x_t = new Tensor(batchSize, channel, z_dims, z_dims, true);
//        
//        latend.showDM("x_start");
//        noise.showDM("noise");
//        
//        iddpm.q_sample(latend, noise, x_t, tx);
//
//        x_t.showDM("x_t");
//        
//        dit.forward(x_t, tx, cx, cos, sin);
//        
//        dit.tensorOP.mul(dit.getOutput(), 10, dit.getOutput());
//        
//        dit.getOutput().showDM();
//        dit.getOutput().showShape();
//
////        Tensor dy = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.order(batchSize * channel * z_dims * z_dims, 0.01f, 0.01f), true);
////        
////        dit.back(dy, cos, sin);
////        
////        dit.update();
////        
////        dit.forward(x, tx, cx, cos, sin);
////        
////        dit.getOutput().showDM();
////        dit.getOutput().showShape();
//       
//        Tensor delta = new Tensor(batchSize, channel * 2, z_dims, z_dims, true);
//        
//        Tensor mean = new Tensor(batchSize, channel, z_dims, z_dims, true);
//        Tensor var = new Tensor(batchSize, channel, z_dims, z_dims, true);
//        dit.tensorOP.getByChannel(dit.getOutput(), mean, 0, 4);
//        dit.tensorOP.getByChannel(dit.getOutput(), var, 4, 4);
//        
//        Tensor vb = iddpm.vb_terms_bpd(mean, var, tx, latend, x_t);
//        
//        vb.showDM("vb");
//        Tensor vbm = new Tensor(1, 1, 1, 1, true);
//        dit.tensorOP.mean(vb, 0, vbm);
//        vbm.showDM("vbm");
//        
//        Tensor dv = iddpm.vb_terms_bpd_back(latend, tx);
//        dv.showDM("dv");
//        
//        dit.loss(mean, noise);
//        
//        Tensor lossDiff = dit.lossDiff(mean, noise);
//        
//        dit.tensorOP.cat(lossDiff, dv, delta);
//        
//        dv.showDMByOffsetRed(0, 32 * 32, "dv");
//        
//        delta.showDMByOffsetRed(4 * 32 * 32, 32 * 32, "delta");
//        
//        dit.back(delta, cos, sin);
//        
//        dit.update();
//        
//        iddpm.q_sample(latend, noise, x_t, tx);
//        
//        dit.forward(x_t, tx, cx, cos, sin);
//        dit.getOutput().showDM();
//        dit.getOutput().showShape();
//        
//        dit.tensorOP.getByChannel(dit.getOutput(), mean, 0, 4);
//        dit.tensorOP.getByChannel(dit.getOutput(), var, 4, 4);
//        
//        vb = iddpm.vb_terms_bpd(mean, var, tx, latend, x_t);
//        
//        vb.showDM("vb");
//        dit.tensorOP.mean(vb, 0, vbm);
//        vbm.showDM("vbm");
//        
//        dv = iddpm.vb_terms_bpd_back(latend, tx);
//        dv.showDM("dv");
//        
//        dit.loss(mean, noise);
//        
//        lossDiff = dit.lossDiff(mean, noise);
//        
//        dit.tensorOP.cat(lossDiff, dv, delta);
//        
//        dv.showDMByOffsetRed(0, 32 * 32, "dv");
//        
//        delta.showDMByOffsetRed(4 * 32 * 32, 32 * 32, "delta");
//        
//        SmoothL1Kernel sk = new SmoothL1Kernel(dit.cudaManager);
//        
//        Tensor a1 = new Tensor(2, 1, 1, 32, MatrixUtils.order(2 * 32, 0.01f, 0.01f), true);
//        Tensor b1 = new Tensor(2, 1, 1, 32, MatrixUtils.order(2 * 32, 0.001f, 0.001f), true);
//        
//        Tensor sl_loss = new Tensor(2, 1, 1, 32, true);
//        
//        sk.forward(a1, b1, sl_loss, 0.05f);
//        
//        sl_loss.showDM();
//        System.err.println(MatrixOperation.sum(sl_loss.syncHost())/sl_loss.dataLength);;
//        
//        sk.backward(a1, b1, sl_loss, 0.05f);
//        
//        sl_loss.showDM();
        
//        Tensor m1 = new Tensor(2, 1, 1, 8, MatrixUtils.order(2 * 8, 0.01f, 0.01f), true);
//        Tensor m2 = new Tensor(2, 1, 1, 8, MatrixUtils.order(2 * 8, 0.02f, 0.02f), true);
//        Tensor l1 = new Tensor(2, 1, 1, 8, MatrixUtils.order(2 * 8, 0.03f, 0.03f), true);
//        Tensor l2 = new Tensor(2, 1, 1, 8, MatrixUtils.order(2 * 8, 0.04f, 0.04f), true);
//        
//        Tensor kl = new Tensor(2, 1, 1, 8, true);
//        
//        iddpm.iddpmKernel.normal_kl(m1, l1, m2, l2, kl);
//        
//        kl.showDM();
        
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
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_Anime(dataLoader, vae, clip);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_pokemon_train() throws Exception {
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 3;
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
        int latendDim = 4;
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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false);
        dit.CUDNN = true;
        dit.learnRate = 0.00005f;
        dit.CLIP_GRAD_NORM = true;
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_Anime(dataLoader, vae, clip);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void getVQVAE32_scale_factor() {
        int batchSize = 8;
        int imageSize = 256;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 1;
        int[] ch_mult = new int[]{1, 2, 2, 4};
        int ch = 32;
        VQVAE2 vae = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\pokemon_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        int[][] indexs = dataLoader.order();
        Tensor out = new Tensor(batchSize, latendDim, z_dims, z_dims, true);
        for (int i = 0; i < indexs.length; i++) {
            System.err.println(i);
            dataLoader.loadData(indexs[i], input);
            JCudaDriver.cuCtxSynchronize();
            Tensor latent = vae.encode(input);
            vae.tensorOP.add(out, latent, out);
        }
        System.err.println("finis sum.");
        Tensor sum = new Tensor(1, 1, 1, 1, true);
        vae.tensorOP.sum(out, sum, 0);
        float meanV = sum.syncHost()[0] / out.dataLength / indexs.length;
        System.err.println(meanV);
        Tensor onceSum = new Tensor(1, 1, 1, 1, true);
        float sum_cpu = 0.0f;
        for (int i = 0; i < indexs.length; i++) {
            System.err.println(i);
            dataLoader.loadData(indexs[i], input);
            JCudaDriver.cuCtxSynchronize();
            Tensor latent = vae.encode(input);
            vae.tensorOP.sub(latent, meanV, latent);
            vae.tensorOP.pow(latent, 2, latent);
            vae.tensorOP.sum(latent, onceSum, 0);
            sum_cpu += onceSum.syncHost()[0];
        }
        System.err.println(sum_cpu);
        System.err.println(sum.syncHost()[0]);
        System.err.println(sum_cpu / sum.syncHost()[0]);
        double scale_factor = Math.sqrt(sum_cpu / out.dataLength / indexs.length);
        System.err.println("scale_factor:" + 1 / scale_factor);
    }
	
	public static void dit_sra_pokemon_train() throws Exception {
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
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
        int z_dims = 32;
        int latendDim = 4;
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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        int block_s = 4;
        int block_t = 8;
        
        boolean qkNorm = false;
        
        DiT_SRA dit = new DiT_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, qkNorm, block_s);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        dit.CLIP_GRAD_NORM = true;
        dit.weight_decay = 0;
        
        DiT_SRA teacher = new DiT_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, qkNorm, block_t);
        teacher.CUDNN = true;
        teacher.RUN_MODEL = RunModel.EVAL;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_SRA_iddpm(dataLoader, vae, clip, iddpm, teacher, "H:\\vae_dataset\\anime_test256\\dit_test2\\", null, 0.13484f);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_iddpm_pokemon_train() throws Exception {
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
        int latendDim = 4;
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
        int depth = 8;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true);
        dit.CUDNN = true;
        dit.learnRate = 0.00005f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_iddpm(dataLoader, vae, clip, iddpm, "H:\\vae_dataset\\anime_test256\\dit_test2\\", "");
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_org_iddpm_pokemon_train() throws Exception {
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
        int latendDim = 4;
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
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "H:\\vae_dataset\\anime_test256\\dit_test2\\", "", 0.13484f);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_org_iddpm_pokemon_cn_train() throws Exception {
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int batchSize = 3;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        int maxContextLen = 64;
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        
        int time = maxContextLen;
        int maxPositionEmbeddingsSize = 512;
        int vocabSize = 21128;
        int hiddenSize = 768;
        int typeVocabSize = 2;
        int headNum = 12;
        int numHiddenLayers = 12;
        int intermediateSize = 3072;
        int textEmbedDim = 512;
        ClipText clip = new ClipText(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, hiddenSize, textEmbedDim, maxPositionEmbeddingsSize, typeVocabSize, intermediateSize, numHiddenLayers);
        clip.CUDNN = true;
        clip.time = time;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "H:\\model\\clip_cn_vit-b-16.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
        
        int z_dims = 32;
        int latendDim = 4;
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
        int depth = 8;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int ditHiddenSize = 384;
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "H:\\vae_dataset\\anime_test256\\dit_test2\\", "", 0.13484f);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_iddpm_amine_train() throws Exception {
        String labelPath = "/omega/dataset/data.json";
        String imgDirPath = "/omega/dataset/256/";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 3;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "/omega/models/bpe_tokenizer/vocab.json";
        String mergesPath = "/omega/models/bpe_tokenizer/merges.txt";
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
        String clipWeight = "/omega/models/clip-vit-base-patch32.json";
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
        String vqvae_model_path = "/omega/models/anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        DiT dit = new DiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.squaredcos, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_iddpm(dataLoader, vae, clip, iddpm, "/omega/test/sd/", "");
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void main(String[] args) {
		 
	        try {
	           
//	        	dit_test();
	        	
//	        	dit_train();
	        	
//	        	dit_pokemon_train();
	        	
//	        	dit_iddpm_pokemon_train();
	        	
//	        	getVQVAE32_scale_factor();
	        	
//	        	dit_sra_pokemon_train();
	        	
//	        	dit_org_iddpm_pokemon_train();
	        	
	        	dit_org_iddpm_pokemon_cn_train();
	        	
	        } catch (Exception e) {
	            // TODO: handle exception
	            e.printStackTrace();
	        } finally {
	            // TODO: handle finally clause
	            CUDAMemoryManager.free();
	        }
	  }
	
}
