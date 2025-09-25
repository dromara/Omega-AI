package com.omega.example.dit.test;

import java.util.Map;
import java.util.Scanner;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.DiTOrgBlock;
import com.omega.engine.nn.layer.dit.DiTSkipBlock;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipText;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.dit.DiT;
import com.omega.engine.nn.network.dit.DiT_ORG;
import com.omega.engine.nn.network.dit.DiT_ORG2;
import com.omega.engine.nn.network.dit.DiT_ORG_SRA;
import com.omega.engine.nn.network.dit.DiT_SRA;
import com.omega.engine.nn.network.dit.DiT_TXT;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.BetaType;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.dit.models.IDDPM;
import com.omega.example.sd.utils.SDImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class DiTTest {
	
	public static void loadWeight(Map<String, Object> weightMap, DiT network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(network.main.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
        ModeLoaderlUtils.loadData(network.main.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear1.weight, weightMap, "y_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear1.bias, weightMap, "y_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear2.weight, weightMap, "y_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear2.bias, weightMap, "y_embedder.mlp.2.bias");
        
        for(int i = 0;i<6;i++){
        	DiTSkipBlock block = network.main.blocks.get(i);
        	block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight");
        	ModeLoaderlUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn1.qL.weight");
            ModeLoaderlUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn1.qL.bias");
        	ModeLoaderlUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn1.kL.weight");
            ModeLoaderlUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn1.kL.bias");
        	ModeLoaderlUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn1.vL.weight");
            ModeLoaderlUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn1.vL.bias");
        	ModeLoaderlUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn1.proj.weight");
            ModeLoaderlUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn1.proj.bias");
        	block.norm2.gamma = ModeLoaderlUtils.loadData(block.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight");
        	ModeLoaderlUtils.loadData(block.modulation.weight, weightMap, "blocks."+i+".default_modulation.1.weight");
            ModeLoaderlUtils.loadData(block.modulation.bias, weightMap, "blocks."+i+".default_modulation.1.bias");
            ModeLoaderlUtils.loadData(block.cross_attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn2.query.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn2.query.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn2.key.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn2.key.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn2.value.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn2.value.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn2.out_proj.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn2.out_proj.bias");
            block.norm3.gamma = ModeLoaderlUtils.loadData(block.norm3.gamma, weightMap, 1, "blocks."+i+".norm3.weight");
            ModeLoaderlUtils.loadData(block.mlp.linear1.weight, weightMap, "blocks."+i+".mlp.fc1.weight");
            ModeLoaderlUtils.loadData(block.mlp.linear1.bias, weightMap, "blocks."+i+".mlp.fc1.bias");
        	ModeLoaderlUtils.loadData(block.mlp.linear2.weight, weightMap, "blocks."+i+".mlp.fc2.weight");
            ModeLoaderlUtils.loadData(block.mlp.linear2.bias, weightMap, "blocks."+i+".mlp.fc2.bias");
            if(block.longSkip) {
            	block.skipNorm.gamma = ModeLoaderlUtils.loadData(block.skipNorm.gamma, weightMap, 1, "blocks."+i+".skip_norm.weight");
            	ModeLoaderlUtils.loadData(block.skipLinear.weight, weightMap, "blocks."+i+".skip_linear.weight");
                ModeLoaderlUtils.loadData(block.skipLinear.bias, weightMap, "blocks."+i+".skip_linear.bias");
            }
        }
        
        network.main.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(network.main.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight");
        
        ModeLoaderlUtils.loadData(network.main.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation1.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation1.bias");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation2.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation2.bias");

    }
	
	public static void loadWeight(Map<String, Object> weightMap, DiT_ORG_SRA network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(network.main.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
        ModeLoaderlUtils.loadData(network.main.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(network.main.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear1.weight, weightMap, "y_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear1.bias, weightMap, "y_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear2.weight, weightMap, "y_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(network.main.labelEmbd.linear2.bias, weightMap, "y_embedder.mlp.2.bias");
        
        for(int i = 0;i<network.depth;i++){
        	DiTOrgBlock block = network.main.blocks.get(i);
        	block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight");
        	block.norm1.beta = ModeLoaderlUtils.loadData(block.norm1.beta, weightMap, 1, "blocks."+i+".norm1.bias");
        	ModeLoaderlUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn1.qL.weight");
            ModeLoaderlUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn1.qL.bias");
        	ModeLoaderlUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn1.kL.weight");
            ModeLoaderlUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn1.kL.bias");
        	ModeLoaderlUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn1.vL.weight");
            ModeLoaderlUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn1.vL.bias");
        	ModeLoaderlUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn1.proj.weight");
            ModeLoaderlUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn1.proj.bias");
        	block.norm2.gamma = ModeLoaderlUtils.loadData(block.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight");
        	block.norm2.beta = ModeLoaderlUtils.loadData(block.norm2.beta, weightMap, 1, "blocks."+i+".norm2.bias");

        	ModeLoaderlUtils.loadData(block.modulation_shift_msa.weight, weightMap, "blocks."+i+".adaLN_modulation_shift_msa.weight");
            ModeLoaderlUtils.loadData(block.modulation_shift_msa.bias, weightMap, "blocks."+i+".adaLN_modulation_shift_msa.bias");
        	ModeLoaderlUtils.loadData(block.modulation_scale_msa.weight, weightMap, "blocks."+i+".adaLN_modulation_scale_msa.weight");
            ModeLoaderlUtils.loadData(block.modulation_scale_msa.bias, weightMap, "blocks."+i+".adaLN_modulation_scale_msa.bias");
        	ModeLoaderlUtils.loadData(block.modulation_gate_msa.weight, weightMap, "blocks."+i+".adaLN_modulation_gate_msa.weight");
            ModeLoaderlUtils.loadData(block.modulation_gate_msa.bias, weightMap, "blocks."+i+".adaLN_modulation_gate_msa.bias");
        	ModeLoaderlUtils.loadData(block.modulation_shift_mlp.weight, weightMap, "blocks."+i+".adaLN_modulation_shift_mlp.weight");
            ModeLoaderlUtils.loadData(block.modulation_shift_mlp.bias, weightMap, "blocks."+i+".adaLN_modulation_shift_mlp.bias");
        	ModeLoaderlUtils.loadData(block.modulation_scale_mlp.weight, weightMap, "blocks."+i+".adaLN_modulation_scale_mlp.weight");
            ModeLoaderlUtils.loadData(block.modulation_scale_mlp.bias, weightMap, "blocks."+i+".adaLN_modulation_scale_mlp.bias");
        	ModeLoaderlUtils.loadData(block.modulation_gate_mlp.weight, weightMap, "blocks."+i+".adaLN_modulation_gate_mlp.weight");
            ModeLoaderlUtils.loadData(block.modulation_gate_mlp.bias, weightMap, "blocks."+i+".adaLN_modulation_gate_mlp.bias");
            
            ModeLoaderlUtils.loadData(block.cross_attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn2.query.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn2.query.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn2.key.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn2.key.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn2.value.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn2.value.bias");
        	ModeLoaderlUtils.loadData(block.cross_attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn2.out_proj.weight");
            ModeLoaderlUtils.loadData(block.cross_attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn2.out_proj.bias");
            block.norm3.gamma = ModeLoaderlUtils.loadData(block.norm3.gamma, weightMap, 1, "blocks."+i+".norm3.weight");
            block.norm3.beta = ModeLoaderlUtils.loadData(block.norm3.beta, weightMap, 1, "blocks."+i+".norm3.bias");
//            ClipModelUtils.loadData(block.mlp.linear1.weight, weightMap, "blocks."+i+".mlp.fc1.weight");
//            ClipModelUtils.loadData(block.mlp.linear1.bias, weightMap, "blocks."+i+".mlp.fc1.bias");
//        	ClipModelUtils.loadData(block.mlp.linear2.weight, weightMap, "blocks."+i+".mlp.fc2.weight");
//            ClipModelUtils.loadData(block.mlp.linear2.bias, weightMap, "blocks."+i+".mlp.fc2.bias");
        }
        
        ModeLoaderlUtils.loadData(network.main.ap_head.linear1.weight, weightMap, "ap_head.linear1.weight");
        ModeLoaderlUtils.loadData(network.main.ap_head.linear1.bias, weightMap, "ap_head.linear1.bias");
        ModeLoaderlUtils.loadData(network.main.ap_head.linear2.weight, weightMap, "ap_head.linear2.weight");
        ModeLoaderlUtils.loadData(network.main.ap_head.linear2.bias, weightMap, "ap_head.linear2.bias");
        
        network.main.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(network.main.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight");
        network.main.finalLayer.finalNorm.beta = ModeLoaderlUtils.loadData(network.main.finalLayer.finalNorm.beta, weightMap, 1, "final_layer.norm_final.bias");
        
        ModeLoaderlUtils.loadData(network.main.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation1.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation1.bias");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation2.weight");
        ModeLoaderlUtils.loadData(network.main.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation2.bias");

    }
	
	public static void dit_test() throws Exception {
		
		int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        int cTime = 77;
        int cDims = 512;
        
        DiT_ORG_SRA dit = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, 4, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, cTime, cDims, mlpRatio, 4, true, false);
        dit.CUDNN = true;
        dit.learnRate=1e-4f;
        dit.CLIP_GRAD_NORM = true;
//        dit.weight_decay = 0;
        
        DiT_ORG_SRA teacher = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, 4, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, cTime, cDims, mlpRatio, 8, true, false);
        teacher.CUDNN = true;
        teacher.RUN_MODEL = RunModel.EVAL;

        String weight = "H:\\model\\dit4.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), dit, true);
        
        int batchSize = 2;
        int channel = 4;
        int z_dims = 32;
        Tensor latend = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.order(batchSize * channel * z_dims * z_dims, -0.03f, 0.01f), true);

        Tensor tx = new Tensor(batchSize, 1, 1, 1, new float[] {0, 853}, true);
        Tensor tt = new Tensor(batchSize, 1, 1, 1, new float[] {1, 653}, true);
       
        Tensor cx = new Tensor(batchSize * cTime, 1, 1, cDims, MatrixUtils.order(batchSize * cTime * cDims, 0.01f, 0.01f), true);
        
        System.out.println(dit.hiddenSize);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(dit.time, dit.hiddenSize, dit.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        cos.showDM("cos");
        sin.showDM("sin");
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        Tensor noise = new Tensor(batchSize, channel, z_dims, z_dims, MatrixUtils.val(batchSize * channel * z_dims * z_dims, 0.1f), true);
        
        Tensor x_t = new Tensor(batchSize, channel, z_dims, z_dims, true);
        
        Tensor tx_t = new Tensor(batchSize, channel, z_dims, z_dims, true);
        
        iddpm.q_sample(latend, noise, x_t, tx);
        iddpm.q_sample(latend, noise, tx_t, tt);
        x_t.showDM();
        tx_t.showDM();
        
        dit.init();
        
        dit.forward(x_t, tx, cx, cos, sin);
        
        dit.getOutput().showDM();
        
        teacher.copyParams(dit);
        
        teacher.forward(x_t, tx, cx, cos, sin);
        
        teacher.getOutput().showDM();
        
        teacher.forward(x_t, tx, cx, cos, sin);
        
        teacher.getOutput().showDM();
        
//        Tensor txr = teacher.getXR();
//        Tensor xr = dit.getXR();
//        Tensor alignLoss = null;
//        Tensor xrDelta = null;
//        if(alignLoss == null || alignLoss.checkShape(xr)) {
//        	alignLoss = Tensor.createGPUTensor(alignLoss, xr.shape(), true);
//        	xrDelta = Tensor.createGPUTensor(xrDelta, xr.shape(), true);
//        }
//        
//        xr.showDM("xr");
//        txr.showDM("txr");
//        
//        dit.smoothL1(xr, txr, alignLoss, 0.05f);
//        dit.smoothL1Back(xr, txr, xrDelta, 0.05f);
//
//        System.err.println(MatrixUtils.sum(alignLoss.syncHost())/alignLoss.dataLength);
//        alignLoss.showDM();

        
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        
        float y_prob = 0.0f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
        
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
        
        float y_prob = 0.0f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 3000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "H:\\vae_dataset\\anime_test256\\dit_test2\\", "", 0.13484f);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_org_sra_pokemon_train() throws Exception {
		String labelPath = "D:\\dataset\\pokemon\\data.json";
        String imgDirPath = "D:\\dataset\\pokemon\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int batchSize = 12;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String tokenizerPath = "D:\\models\\clip\\clip_cn\\vocab.txt";
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
        String clipWeight = "D:\\models\\clip\\clip_cn\\clip_cn_vit-b-16.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
	        
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
        String vqvae_model_path = "D:\\models\\pokemon_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int ditHiddenSize = 1024;
        
        int block_s = 4;
        int block_t = 8;
        
        boolean qkNorm = false;
        
        DiT_ORG_SRA dit = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, block_s, true, qkNorm);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        dit.CLIP_GRAD_NORM = true;
//        dit.weight_decay = 0;
        
        DiT_ORG_SRA teacher = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, block_t, true, qkNorm);
        teacher.CUDNN = true;
        teacher.RUN_MODEL = RunModel.EVAL;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 3000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_SRA_iddpm(dataLoader, vae, clip, iddpm, teacher, "D:\\dataset\\dit_test3\\", null, 0.13484f);
//        String save_model_path = "/omega/models/sd_anime256.model";
//        ModelUtils.saveModel(unet, save_model_path);
    }
	
	public static void dit_org_sra_amine_train() throws Exception {
		String labelPath = "/omega/dataset/data.json";
        String imgDirPath = "/omega/dataset/256/";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 8;
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int ditHiddenSize = 1024;
        
        int block_s = 4;
        int block_t = 8;
        
        boolean qkNorm = false;
        
        DiT_ORG_SRA dit = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, block_s, true, qkNorm);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        dit.CLIP_GRAD_NORM = true;
        
        String model_path = "/omega/models/anime_dit_700.model";
        ModelUtils.loadModel(dit, model_path);
        
        DiT_ORG_SRA teacher = new DiT_ORG_SRA(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, block_t, true, qkNorm);
        teacher.CUDNN = true;
        teacher.RUN_MODEL = RunModel.EVAL;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        
        optimizer.train_DiT_ORG_SRA_iddpm(dataLoader, vae, clip, iddpm, teacher, "/omega/test/dit/", "/omega/models/dit/", 0.18215f);
        String save_model_path = "/omega/models/dit_sra_anime256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void dit_iddpm_amine_train() throws Exception {
        String labelPath = "/omega/dataset/data.json";
        String imgDirPath = "/omega/dataset/256/";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 8;
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
	
	public static void dit_org_iddpm_amine_train() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 16;
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        String vqvae_model_path = "D:\\models\\anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        float y_prob = 0.0f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
//        String model_path = "D:\\models\\dit_anime256.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "D:\\test\\dit\\", "D:\\models\\dit\\", 0.18125f);
        String save_model_path = "D:\\models\\dit\\dit_anime_384_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void dit_org_iddpm_100K_train() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

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
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
//        String model_path = "D:\\models\\dit_anime256.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "D:\\test\\dit\\", "D:\\models\\dit\\", 0.18125f);
        String save_model_path = "D:\\models\\dit\\dit_anime_384_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
    
	public static void dit_org_iddpm_amine_train2() throws Exception {
		String labelPath = "/omega/dataset/data.json";
        String imgDirPath = "/omega/dataset/256/";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 10;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "/omega/models/vocab.json";
        String mergesPath = "/omega/models/merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "/omega/models/sdxl-vae-fp16-fix.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
//        String model_path = "D:\\models\\dit_anime256.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "/omega/test/dit/", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void dit_org_iddpm_amine_train3() throws Exception {
		String labelPath = "D:\\dataset\\amine\\data.json";
        String imgDirPath = "D:\\dataset\\amine\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 16;
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

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
        
        int ditHeadNum = 6;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 384;
        
        float y_prob = 0.1f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
//        String model_path = "D:\\models\\dit_anime256.model";
//        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.train_DiT_ORG_iddpm(dataLoader, vae, clip, iddpm, "D://test//dit//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void dit_org_iddpm_amine_train4() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);

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
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.0f;
        
        DiT_ORG2 dit = new DiT_ORG2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 1e-4f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        optimizer.train_DiT_ORG_iddpm_no_kl(dataLoader, vae, clip, iddpm, "D://test//dit2//", "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_anime_768_256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void dit_org_sra_amine_train2() throws Exception {
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
        
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
        
        int ditHeadNum = 12;
        int latendSize = 32;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int ditHiddenSize = 768;
        
        float y_prob = 0.0f;
        
        int block_s = 4;
        int block_t = 8;
        
        DiT_ORG2 dit = new DiT_ORG2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, y_prob, block_s);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        dit.CLIP_GRAD_NORM = true;
        
//        String model_path = "/omega/models/anime_dit_700.model";
//        ModelUtils.loadModel(dit, model_path);
        
        DiT_ORG2 teacher = new DiT_ORG2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, ditHiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, false, y_prob, block_t);
        teacher.CUDNN = true;
        teacher.RUN_MODEL = RunModel.EVAL;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        
        optimizer.train_DiT_ORG_SRA_iddpm(dataLoader, vae, clip, iddpm, teacher, "D:\\test\\dit_sra\\", "/omega/models/dit/", 0.18215f);
        String save_model_path = "/omega/models/dit_sra_anime256.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	
	
	public static void dit_org_iddpm_amine_predict() throws Exception {

        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 1;
        float[] imgMean = new float[]{0.5f, 0.5f, 0.5f};
        float[] imgStd = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
       
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
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, false);
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
        String vqvae_model_path = "D:\\models\\anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vqvae_model_path);
        
        int ditHeadNum = 16;
        int latendSize = 32;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        float y_prob = 0;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        dit.RUN_MODEL = RunModel.TEST;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);
        
        String model_path = "D:\\test\\models\\mmdit\\dit_xl2_6.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(dit.time, dit.hiddenSize, dit.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];

        Tensor t = new Tensor(batchSize, 1, 1, 1, true);

        Tensor noise = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        
        Tensor mean = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        Tensor var = new Tensor(batchSize, dit.inChannel, dit.height, dit.width, true);
        
        Tensor condInput = null;
        Tensor latend = new Tensor(batchSize, latendDim, 32, 32, true);
        
        try {
            
        	Tensor label = new Tensor(batchSize * dit.maxContextLen, 1, 1, 1, true);
            Scanner scanner = new Scanner(System.in);
        	while (true) {
                 System.out.println("请输入英文提示词:");
                 String input_txt = scanner.nextLine();
                 if (input_txt.equals("exit")) {
                     break;
                 }
                 input_txt = input_txt.toLowerCase();
                 //			System.out.println(text);
                 int[] ids = bpe.encodeInt(input_txt, maxContextLen);
                 for (int j = 0; j < maxContextLen; j++) {
                     if (j < ids.length) {
                         label.data[j] = ids[j];
                     } else {
                         label.data[j] = 0;
                     }
                 }
                 
                 label.hostToDevice();
                 label.showDM();
                 /**
                  * get context embd
                  */
                 condInput = clip.forward(label);
                 
                 RandomUtils.gaussianRandom(noise, 0, 1);
                 
                 Tensor sample = iddpm.p_sample(dit, cos, sin, latend, noise, condInput, t, mean, var);
                 
                 JCuda.cudaDeviceSynchronize();

                 dit.tensorOP.mul(sample, 1.0f / 0.18125f, sample);

                 Tensor result = vae.decode(sample);
                 JCuda.cudaDeviceSynchronize();
                 result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
                 //			System.err.println("in");
                 /**
                  * print image
                  */
                 showImgs("D://test//dit/", result, imgMean, imgStd);
        	}
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }

    }
	
    public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }
    
	public static void dit_xl2_iddpm_train() throws Exception {
//		String dataPath = "/omega/dataset/txt2img_latend.bin";
//        String clipDataPath = "/omega/dataset/txt2img_clip.bin";
		String dataPath = "D:\\dataset\\amine\\amine_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\amine_clip.bin";

        int batchSize = 10;
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
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContext, textEmbedDim, mlpRatio, true, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0001f;
        
        IDDPM iddpm = new IDDPM(timeSteps, BetaType.linear, dit.cudaManager);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

        optimizer.train_DiT_ORG_iddpm(dataLoader, iddpm, "/omega/models/dit/", 0.13025f);
        String save_model_path = "/omega/models/dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
    }
	
	public static void complate_ms() throws Exception {

		String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_clip.bin";

        int batchSize = 1000;
        int latendDim = 4;
        int height = 32;
        int width = 32;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int[][] indexs = dataLoader.shuffle();

        Tensor latend = new Tensor(batchSize, dataLoader.channel, dataLoader.height, dataLoader.width, true);
        Tensor condInput = new Tensor(batchSize , 1, 1, dataLoader.clipEmbd, true);
        
        int count = 10000;
        
        int tmp = count * height * width - 1;
        
        float[] mean = new float[] {0.0f, 0.0f, 0.0f, 0.0f};
        float[] std = new float[] {0.0f, 0.0f, 0.0f, 0.0f};
        
//        int N2 = 2;
//        int C2 = 3;
//        int H2 = 2;
//        int W2 = 2;
//        int len = N2 * C2 * H2 * W2;
//        float[] data = RandomUtils.order(len, 1.0f, 1.0f);
//        float[] mean3 = new float[3];
//        float[] mean4 = new float[3];
//        for(int i = 0;i<len;i++) {
//        	int c = i / H2 / W2 % C2;
//        	mean3[c] += data[i] / (N2 * H2 * W2);
//        	mean4[c] += data[i] / (N2 * H2 * W2 - 1);
//        }
//        System.out.println(JsonUtils.toJson(data));
//        System.out.println(JsonUtils.toJson(mean3));
//        
//        float[] std4 = new float[3];
//        for(int i = 0;i<len;i++) {
//        	int c = i / H2 / W2 % C2;
//        	std4[c] += Math.pow(data[i] - mean3[c], 2) / (N2 * H2 * W2 - 1);
//        }
//        for(int c = 0;c<C2;c++) {
//        	std4[c] = (float) Math.sqrt(std4[c]);
//        }
//        System.out.println(JsonUtils.toJson(std4));
        /**
         * 遍历整个训练集
         */
        for (int it = 0; it < 10; it++) {
            System.out.println("mean:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
            	mean[c] += latend.data[i] / (tmp + 1);
            }

        }
        System.out.println("mean finish.");
        System.out.println(JsonUtils.toJson(mean));
        for (int it = 0; it < 10; it++) {
            System.out.println("std:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
            	std[c] += Math.pow(latend.data[i] - mean[c], 2) / tmp;
            }

        }
        
        for(int c = 0;c<4;c++) {
        	std[c] = (float) Math.sqrt(std[c]);
        }
        
        System.out.println(JsonUtils.toJson(mean));
        System.out.println(JsonUtils.toJson(std));
	}
	
	public static void complate_ms2() throws Exception {

		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_vavae_clip.bin";

        int batchSize = 1000;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int[][] indexs = dataLoader.shuffle();

        Tensor latend = new Tensor(batchSize, dataLoader.channel, dataLoader.height, dataLoader.width, true);
        Tensor condInput = new Tensor(batchSize , 1, 1, dataLoader.clipEmbd, true);
        
        int count = 10000;
        
        int tmp = count * height * width - 1;
        
        float[] mean = RandomUtils.val(latendDim, 0.0f);
        float[] std = RandomUtils.val(latendDim, 0.0f);

        /**
         * 遍历整个训练集
         */
        for (int it = 0; it < 10; it++) {
            System.out.println("mean:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
            	mean[c] += latend.data[i] / (tmp + 1);
            }

        }
        System.out.println("mean finish.");
        System.out.println(JsonUtils.toJson(mean));
        for (int it = 0; it < 10; it++) {
            System.out.println("std:"+it);
            dataLoader.loadData(indexs[it], latend, condInput, it);
            
            for(int i = 0;i<latend.dataLength;i++) {
            	int c = i / latend.height / latend.width % latend.channel;
            	std[c] += Math.pow(latend.data[i] - mean[c], 2) / tmp;
            }

        }
        
        for(int c = 0;c<latendDim;c++) {
        	std[c] = (float) Math.sqrt(std[c]);
        }
        
        System.out.println(JsonUtils.toJson(mean));
        System.out.println(JsonUtils.toJson(std));
	}
	
	public static void dit_b2_iddpm_train() throws Exception {
//		String dataPath = "/omega/dataset/txt2img_latend.bin";
//        String clipDataPath = "/omega/dataset/txt2img_clip.bin";
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_vavae_clip.bin";

        int batchSize = 128;
        int latendDim = 32;
        int height = 16;
        int width = 16;
        int textEmbedDim = 768;
        int maxContext = 1;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int ditHeadNum = 12;
        int latendSize = 16;
        int depth = 12;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.3f;
        
        DiT_ORG dit = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, maxContext, textEmbedDim, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 200, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
        optimizer.train_DiT_ICPlan(dataLoader, icplan, "D://models//", mean, std, 1f, 2);
        String save_model_path = "/omega/models/dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
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
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.3f;
        
        DiT_ORG network = new DiT_ORG(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, 1, textEmbedDim, mlpRatio, true, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        network.RUN_MODEL = RunModel.TEST;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\models\\dit_xl2\\dit_xl2_18.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
        
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

            RandomUtils.gaussianRandom(noise, 0, 1);
            
            Tensor sample = icplan.sample(network, noise, t, condInput, cos, sin, latend);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit4\\dit_org\\" + i, result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void dit_txt_b2_iddpm_train() throws Exception {
		String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
        String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";

        int batchSize = 100;
        int latendDim = 32;
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
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        DiT_TXT dit = new DiT_TXT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, false, y_prob);
        dit.CUDNN = true;
        dit.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);

        String model_path = "D:\\models\\dit_txt\\dit_b2_700.model";
        ModelUtils.loadModel(dit, model_path);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 1000, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);
        
        optimizer.train_DiT_TXT_ICPlan(dataLoader, icplan, "D://models//dit_txt//", mean, std, 1f, 20);
        String save_model_path = "/omega/models/dit_xl2.model";
        ModelUtils.saveModel(dit, save_model_path);
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
        int patchSize = 2;
        int hiddenSize = 768;
        
        float y_prob = 0.1f;
        
        DiT_TXT network = new DiT_TXT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\models\\dit_txt\\dit_b2_300.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);

        Tensor condInput = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[10];
        for(int i = 0;i<4;i++) {
        	System.out.println("start create test images.");
            labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed.";
            labels[3] = "a little girl standing on the beach";
            labels[4] = "fruit cream cake";
            labels[5] = "A cheerful cartoon character swings on a swing in a serene, colorful outdoor setting.";
            
            labels[6] = "A woman in a red, futuristic ensemble wields a firearm amidst a bustling cityscape with mixed architecture, where mythical creatures fly overhead while a misty atmosphere surrounds the scene, accompanied by neon-lit characters written in a Chinese script.";
            
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
            condInput = clip.get_full_clip_prompt_embeds(label);

            RandomUtils.gaussianRandom(noise, 0, 1);
            
            Tensor sample = icplan.sample(network, noise, t, condInput, cos, sin, latend);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\dit_1\\" + i, result, mean, std);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_rope3() throws Exception {
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
        
        int ditHeadNum = 16;
        int latendSize = 16;
        int depth = 24;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 2;
        int hiddenSize = 1024;
        
        float y_prob = 0.1f;
        
        DiT_TXT network = new DiT_TXT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxPositionEmbeddingsSize, mlpRatio, false, y_prob);
        network.CUDNN = true;
        network.learnRate = 0.0002f;
        
        ICPlan icplan = new ICPlan(network.tensorOP);
        
        String model_path = "D:\\test\\models\\dit_xl2\\dit_b2_2.model";
        ModelUtils.loadModel(network, model_path);
        
        Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        Tensor latend = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(network.time, network.hiddenSize, network.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        Tensor latendMean = new Tensor(latendDim, 1, 1, 1, new float[] {0.23869862f,0.4016211f,-0.15087046f,-0.52679396f,-0.15986611f,-1.6260003f,-0.5108059f,0.036283042f,0.3879915f,0.5334558f,-0.96909237f,1.4872372f,0.071545064f,0.7708449f,0.16623285f,0.7733368f,-0.9222466f,1.2859207f,-0.30753133f,-0.70088845f,0.5247328f,0.09425582f,-1.1671793f,0.53027356f,2.7668183f,1.4706479f,0.09313846f,-0.25821307f,-0.81280077f,-0.56423014f,0.49580055f,-0.35338005f}, true);
        Tensor latendStd = new Tensor(latendDim, 1, 1, 1, new float[] {4.1767454f,4.245004f,3.4222624f,3.6970704f,3.6395364f,3.3921142f,3.0486407f,3.6789029f,3.922576f,3.760961f,3.7205217f,3.70206f,3.7118554f,3.6425886f,3.223105f,3.3205664f,4.135744f,3.6481087f,3.6758296f,3.0634696f,3.3749795f,2.9729145f,3.8634508f,4.518134f,2.7782023f,3.4923503f,4.7507596f,3.2647762f,3.3624852f,3.7219477f,4.659944f,4.2925563f}, true);

        network.RUN_MODEL = RunModel.TEST;
        String[] labels = new String[10];
        for(int i = 0;i<10;i++) {
        	System.out.println("start create test images.");
            labels[0] = "A cat holding a sign that says hello world";
            labels[1] = "a vibrant anime mountain lands";
            labels[2] = "a highly detailed anime landscape,big tree on the water, epic sky,golden grass,detailed";
            labels[3] = "a little girl standing on the beach";
            labels[4] = "fruit cream cake";
            labels[5] = "a yellow apple is placed on the plate";
            
	        labels[6] = "A corgi is taking a walk under the sea";
            labels[7] = "A woman with shoulder-length blonde hair wearing a dark blouse with a floral patterned collar";
            labels[8] = "A small, grey crochet plush toy of a cat with pink paws and a pink nose sits on a wooden surface";
            labels[9] = "A group of humpback whales is swimming in the ocean, with one whale prominently in the foreground and two others in the background. The water is clear, and the whales are surrounded by a multitude of bubbles, creating a dynamic underwater scene";
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

            RandomUtils.gaussianRandom(noise, 0, 1);
            
            Tensor sample = icplan.sample(network, noise, t, condInput, cos, sin, latend);
            
            icplan.latend_un_norm(sample, latendMean, latendStd);

            Tensor result = vae.decode(sample);
            
            JCuda.cudaDeviceSynchronize();
            
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);

            showImgs("D:\\test\\dit_vavae\\dit_3\\" + i, result, mean, std);
            
            System.out.println("finish create.");
        }
        
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
	        	
//	        	dit_org_iddpm_pokemon_cn_train();
	        	
//	        	dit_org_iddpm_amine_train();
	        	
//	        	dit_org_sra_pokemon_train();
	        	
//	        	dit_org_sra_amine_train();
	        	
//	        	dit_org_iddpm_amine_predict();
	        	
//	        	dit_org_iddpm_100K_train();
	        	
//	        	dit_org_iddpm_amine_train3();
	        	
//	        	dit_org_iddpm_amine_train4();
	        	
//	        	dit_org_sra_amine_train2();
	        	
//	        	dit_xl2_iddpm_train();
	        	
//	        	dit_b2_iddpm_train();
	        	
//	        	complate_ms();
	        	
//	        	complate_ms2();
	        	
//	        	test_rope();
	        	
//	        	dit_b2_iddpm_train();
	        	
//	        	dit_txt_b2_iddpm_train();
	        	
//	        	test_rope2();
	        	
	        	test_rope3();

	        } catch (Exception e) {
	            // TODO: handle exception
	            e.printStackTrace();
	        } finally {
	            // TODO: handle finally clause
	            CUDAMemoryManager.free();
	        }
	  }
	
}
