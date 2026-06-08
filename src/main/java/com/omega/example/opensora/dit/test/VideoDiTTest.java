package com.omega.example.opensora.dit.test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.video.rope.RoPE3DKernel;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.LTXVideo_VAE;
import com.omega.engine.nn.network.vae.LTXVideo_VAE_Decoder;
import com.omega.engine.nn.network.vae.LTXVideo_VAE_Encoder;
import com.omega.engine.nn.network.video.OmegaVideo;
import com.omega.engine.nn.network.video.OmegaVideo2;
import com.omega.engine.nn.network.video.OmegaVideoI2V;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.opensora.dataset.LatendDatasetI2V;
import com.omega.example.opensora.utils.VideoReaderExample;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

public class VideoDiTTest {
	
	public static void complate_ms() throws Exception {

		String dataPath = "D:\\dataset\\video\\1w_latend.bin";
		String clipDataPath = "D:\\dataset\\video\\1w_clip.bin";

        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        
        int batchSize = 1000;
        int latendDim = 128;
        int num_frames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * num_frames, height, width, maxContext, textEmbedDim, BinDataType.float32);
        
        int[][] indexs = dataLoader.shuffle();

        Tensor latend = new Tensor(batchSize, latendDim, num_frames * dataLoader.height, dataLoader.width, true);
        Tensor condInput = new Tensor(batchSize * dataLoader.clipMaxTime, 1, 1, dataLoader.clipEmbd, true);
        
        int count = 10000;
        
        int tmp = count * num_frames * height * width - 1;
        
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
//            	System.out.println(latend.height + ":" + latend.channel);
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
        
        Tensor mean_t = new Tensor(latendDim, 1, 1, 1, mean, true);
        Tensor std_t = new Tensor(latendDim, 1, 1, 1, std, true);
        
        File file = new File(meanStdPath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
        	com.omega.engine.nn.network.utils.ModelUtils.saveParams(rFile, mean_t);
        	com.omega.engine.nn.network.utils.ModelUtils.saveParams(rFile, std_t);
            System.out.println("save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }

	}
	
	public static void video_dit_train() throws Exception {
		
		String dataPath = "D:\\dataset\\video\\1w_latend.bin";
		String clipDataPath = "D:\\dataset\\video\\1w_clip.bin";
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        
        int batchSize = 6;
        int latendDim = 128;
        int numFrames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * numFrames, height, width, maxContext, textEmbedDim, BinDataType.float32);
		
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        int maxContextLen = 77;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaVideo dit = new OmegaVideo(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, height, width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
         
//        String model_path = "D:\\dataset\\video\\models\\video_dit_b_44.model";
//        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        
        loadMS(meanStdPath, mean, std);

        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        optimizer.train_video_dit_ICPlan(dataLoader, icplan, "D:\\dataset\\video\\models\\video_dit_b_", mean, std, 1);
        
        String save_model_path = "D:\\dataset\\video\\models\\video_dit_b.model";
        ModelUtils.saveModel(dit, save_model_path);
	}
	
	
	public static void video_dit_test() throws Exception {
		
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
		
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxPositionEmbeddingsSize, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxPositionEmbeddingsSize;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

		int num_frames = 17;
		int height = 352;
		int width = 640;
		int patch_size_t = 1;
		int patch_size = 4;
		int[] block_out_channels = new int[] {128, 256, 512, 512};
		int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
		boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
		
		LTXVideo_VAE vae = new LTXVideo_VAE(LossType.MSE, UpdaterType.adamw, num_frames, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		vae.CUDNN = true;
		String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
        ModelUtils.loadModel(vae, save_model_path);
        
        int latendDim = 128;
        int vae_numFrames = 3;
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        int maxContextLen = 77;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;

        int vae_height = 11;
        int vae_width = 20;
        OmegaVideo dit = new OmegaVideo(LossType.MSE, UpdaterType.adamw, latendDim, vae_numFrames, vae_height, vae_width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        String model_path = "D:\\dataset\\video\\models\\video_dit_b_44.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        loadMS(meanStdPath, mean, std);
        
        int batchSize = 2;
        
        int thw = vae_numFrames * vae_height * vae_width;
        
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor latend = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor eps = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor noise2 = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor video = new Tensor(batchSize * num_frames, 3, height, width, true);
        
        Tensor[][] cs = RoPE3DKernel.init3DRoPE(vae_numFrames, vae_height, vae_width, hiddenSize, ditHeadNum, 1f, 1.4375f, 2.5f);
        Tensor[] cos = cs[0];
        Tensor[] sin = cs[1];

        dit.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "The video features a man in a suit, looking off to the side with a serious expression. The man is the main subject of the video, and he is dressed in a dark suit with a light-colored shirt and tie. The background is blurred, but it appears to be an indoor setting with other people present. The lighting is soft and natural, suggesting an indoor environment with large windows or skylights. The man's expression and the setting suggest a serious or professional context. ";
        labels[1] = "The video features a young man with curly hair";
//        labels[2] = "The video features a young man with curly hair";
//        labels[3] = "The video features a young man with curly hair";

        SDImageDataLoaderEN.loadLabel_offset(label, 0, labels[0], bpe, maxContextLen);
        SDImageDataLoaderEN.loadLabel_offset(label, 1, labels[1], bpe, maxContextLen);
//        SDImageDataLoaderEN.loadLabel_offset(label, 2, labels[2], bpe, maxContextLen);
//        SDImageDataLoaderEN.loadLabel_offset(label, 3, labels[3], bpe, maxContextLen);
        
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
            Tensor y_null = dit.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	dit.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
            }
        }
        
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A cat holding a sign that says hello world";
                labels[1] = "A fox sleeping inside a large tansparent lightbule";
  
                SDImageDataLoaderEN.loadLabel_offset(label, 0, labels[0], bpe, maxContextLen);
                SDImageDataLoaderEN.loadLabel_offset(label, 1, labels[1], bpe, maxContextLen);

                condInput = clip.get_full_clip_prompt_embeds(label);
        	}
        	
        	System.out.println("start create test videos.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg(dit, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

            icplan.latend_un_norm(sample, mean, std, thw);

            Tensor result = vae.decode(sample);
//          result.showShape("result");
            vae.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\"+i);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(dit, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);
            
            icplan.latend_un_norm(sample, mean, std, thw);

            result = vae.decode(sample);
            vae.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\t_"+i);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void video_dit_train2() throws Exception {
		
		String dataPath = "D:\\dataset\\video\\1w_latend.bin";
		String clipDataPath = "D:\\dataset\\video\\1w_clip.bin";
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        
        int batchSize = 4;
        int latendDim = 128;
        int numFrames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDataset dataLoader = new LatendDataset(dataPath, clipDataPath, batchSize, latendDim * numFrames, height, width, maxContext, textEmbedDim, BinDataType.float32);
		
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        int maxContextLen = 77;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaVideo2 dit = new OmegaVideo2(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, height, width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
         
//        String model_path = "D:\\dataset\\video\\models\\video_dit_b_16.model";
//        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        
        loadMS(meanStdPath, mean, std);

        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        optimizer.train_video_dit_ICPlan2(dataLoader, icplan, "D:\\dataset\\video\\models\\video_dit_b_", mean, std, 2);
        
        String save_model_path = "D:\\dataset\\video\\models\\video_dit_b.model";
        ModelUtils.saveModel(dit, save_model_path);
	}
	
	public static void video_dit_test2() throws Exception {
		
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
		
        int maxPositionEmbeddingsSize = 77;
        int vocabSize = 49408;
        int headNum = 12;
        int n_layers = 12;
        int textEmbedDim = 768;
        int intermediateSize = 3072;
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxPositionEmbeddingsSize, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxPositionEmbeddingsSize;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

		int num_frames = 17;
		int height = 352;
		int width = 640;
		int patch_size_t = 1;
		int patch_size = 4;
		int[] block_out_channels = new int[] {128, 256, 512, 512};
		int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
		boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
		
		LTXVideo_VAE vae = new LTXVideo_VAE(LossType.MSE, UpdaterType.adamw, num_frames, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		vae.CUDNN = true;
		String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
        ModelUtils.loadModel(vae, save_model_path);
        
        int latendDim = 128;
        int vae_numFrames = 3;
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int mlpRatio = 4;
        int patchSize = 1;
        int hiddenSize = 1152;
        int maxContextLen = 77;
        
        float y_prob = 0.1f;
        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;

        int vae_height = 11;
        int vae_width = 20;
        OmegaVideo2 dit = new OmegaVideo2(LossType.MSE, UpdaterType.adamw, latendDim, vae_numFrames, vae_height, vae_width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContextLen, mlpRatio, token_drop, path_drop_prob, y_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        String model_path = "D:\\dataset\\video\\models\\video_dit_b.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        loadMS(meanStdPath, mean, std);
        
        int batchSize = 2;
        
        int thw = vae_numFrames * vae_height * vae_width;
        
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
       
        Tensor condInput = null;
        Tensor condInput_ynull = null;
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor latend = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor eps = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor noise2 = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor video = new Tensor(batchSize * num_frames, 3, height, width, true);
        
        Tensor[][] cs = RoPE3DKernel.init3DRoPE(vae_numFrames, vae_height, vae_width, hiddenSize, ditHeadNum, 1f, 1.4375f, 2.5f);
        Tensor[] cos = cs[0];
        Tensor[] sin = cs[1];

        dit.RUN_MODEL = RunModel.EVAL;
        String[] labels = new String[batchSize];
        labels[0] = "The video features a man in a suit, looking off to the side with a serious expression. The man is the main subject of the video, and he is dressed in a dark suit with a light-colored shirt and tie. The background is blurred, but it appears to be an indoor setting with other people present. The lighting is soft and natural, suggesting an indoor environment with large windows or skylights. The man's expression and the setting suggest a serious or professional context. ";
        labels[1] = "The video features a man with a beard and short hair, wearing a brown shirt.";
//        labels[2] = "The video features a young man with curly hair";
//        labels[3] = "The video features a young man with curly hair";

        SDImageDataLoaderEN.loadLabel_offset(label, 0, labels[0], bpe, maxContextLen);
        SDImageDataLoaderEN.loadLabel_offset(label, 1, labels[1], bpe, maxContextLen);
//        SDImageDataLoaderEN.loadLabel_offset(label, 2, labels[2], bpe, maxContextLen);
//        SDImageDataLoaderEN.loadLabel_offset(label, 3, labels[3], bpe, maxContextLen);
        
        condInput = clip.get_full_clip_prompt_embeds(label);
        
        if(condInput_ynull == null) {
        	condInput_ynull = Tensor.createGPUTensor(condInput_ynull, condInput.number, condInput.channel, condInput.height, condInput.width, true);
            Tensor y_null = dit.main.labelEmbd.getY_embedding();
            int part_input_size = y_null.dataLength;
            for(int b = 0;b<batchSize;b++) {
            	dit.tensorOP.op.copy_gpu(y_null, condInput_ynull, part_input_size, 0, 1, b * part_input_size, 1);
            }
        }
        
        for(int i = 0;i<10;i++) {
        	
        	if(i > 4) {
        		labels[0] = "A long hair girl";
                labels[1] = "A old man";
  
                SDImageDataLoaderEN.loadLabel_offset(label, 0, labels[0], bpe, maxContextLen);
                SDImageDataLoaderEN.loadLabel_offset(label, 1, labels[1], bpe, maxContextLen);

                condInput = clip.get_full_clip_prompt_embeds(label);
        	}
        	
        	System.out.println("start create test videos.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg(dit, noise, t, condInput, condInput_ynull, cos, sin, latend, eps, 1.0f);

            icplan.latend_un_norm(sample, mean, std, thw);

            Tensor result = vae.decode(sample);
//          result.showShape("result");
            vae.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\"+i);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(dit, noise2, t, condInput, condInput_ynull, cos, sin, latend, eps, 2.0f);
            
            icplan.latend_un_norm(sample, mean, std, thw);

            result = vae.decode(sample);
            vae.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\t_"+i);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void test_vae_img() throws Exception {
			
		String dataPath = "D:\\dataset\\video\\1w_latend.bin";
		String imgDataPath = "D:\\dataset\\video\\1W_img_latend.bin";
		String clipDataPath = "D:\\dataset\\video\\1w_clip.bin";
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        
        int batchSize = 2;
        int latendDim = 128;
        int numFrames = 3;
        int vae_height = 11;
        int vae_width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDatasetI2V dataLoader = new LatendDatasetI2V(dataPath, imgDataPath, clipDataPath, batchSize, latendDim, numFrames, vae_height, vae_width, maxContext, textEmbedDim, BinDataType.float32);

        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        
        loadMS(meanStdPath, mean, std);

		int num_frames = 17;
		int height = 352;
		int width = 640;
		int patch_size_t = 1;
		int patch_size = 4;
        int vae_numFrames = 3;
		int[] block_out_channels = new int[] {128, 256, 512, 512};
		int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
		boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
		
		LTXVideo_VAE_Encoder vae_encoder = new LTXVideo_VAE_Encoder(LossType.MSE, UpdaterType.adamw, 1, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		LTXVideo_VAE_Decoder vae_decoder = new LTXVideo_VAE_Decoder(LossType.MSE, UpdaterType.adamw, vae_numFrames, vae_height, vae_width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		vae_encoder.CUDNN = true;
		vae_decoder.CUDNN = true;
		String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
        ModelUtils.loadModel(vae_encoder, vae_decoder, save_model_path);
        
        Tensor latend = new Tensor(batchSize, latendDim * vae_numFrames, vae_height, vae_width, true);
        Tensor img_latend = new Tensor(batchSize, latendDim, vae_height, vae_width, true);
        
        Tensor condInput = new Tensor(batchSize * 77, 1, 1, textEmbedDim, true);
        
        int[][] indexs = dataLoader.order();
        
        int[] next = indexs[0];
        dataLoader.loadData(indexs[2], next, latend, img_latend, condInput, 2);
        
        ICPlan icplan = new ICPlan(vae_encoder.tensorOP);
        
        int thw = vae_numFrames * vae_height * vae_width;

        icplan.latend_norm(latend, mean, std, thw);    
        icplan.latend_norm(img_latend, mean, std, 1 * vae_height * vae_width);    
//        float scaling_factor = 0.41407f;
//        vae_decoder.tensorOP.mul(latend, scaling_factor, latend);
//        vae_decoder.tensorOP.mul(img_latend, scaling_factor, img_latend);
        
        vae_encoder.tensorOP.getByChannel_back(latend, img_latend, new int[] {batchSize * latendDim, vae_numFrames, vae_height, vae_width}, 0, 1);

        Tensor video = new Tensor(batchSize * num_frames, 3, height, width, true);
        
//        vae_decoder.tensorOP.div(latend, scaling_factor, latend);
        icplan.latend_un_norm(latend, mean, std, thw);

        Tensor result = vae_decoder.decode(latend);
        vae_decoder.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

        video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

        tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\t_");
	}
	
	public static void test_img() throws Exception {
		

        int batchSize = 2;
        int vae_height = 11;
        int vae_width = 20;
		int height = 352;
		int width = 640;
		int patch_size_t = 1;
		int patch_size = 4;
        int vae_numFrames = 1;
		int[] block_out_channels = new int[] {128, 256, 512, 512};
		int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
		boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
		
		LTXVideo_VAE_Encoder vae_encoder = new LTXVideo_VAE_Encoder(LossType.MSE, UpdaterType.adamw, 1, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		LTXVideo_VAE_Decoder vae_decoder = new LTXVideo_VAE_Decoder(LossType.MSE, UpdaterType.adamw, vae_numFrames, vae_height, vae_width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		vae_encoder.CUDNN = true;
		vae_decoder.CUDNN = true;
		String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
        ModelUtils.loadModel(vae_encoder, vae_decoder, save_model_path);

        String[] imgPaths = new String[] {
        		"C:\\Users\\Administrator\\Desktop\\test_video\\ScreenShot_2026-05-21_165922_765.png",
        		"C:\\Users\\Administrator\\Desktop\\test_video\\ScreenShot_2026-05-26_184055_227.png"
        };
        
        Tensor img_input = new Tensor(batchSize, 3, height, width, true);
        
        VideoReaderExample.loadVImg2Tensor(imgPaths[0], 1, height, width, img_input, 0);
        VideoReaderExample.loadVImg2Tensor(imgPaths[1], 1, height, width, img_input, 1);
        
        img_input.hostToDevice();
        
        Tensor img_latend = vae_encoder.encode(img_input);
        
        Tensor video = new Tensor(batchSize * 1, 3, height, width, true);

        Tensor result = vae_decoder.decode(img_latend);
        vae_decoder.tensorOP.permute(result, video, new int[] {batchSize, 3, 1, height, width}, new int[] {batchSize, 1, 3, height, width}, new int[] {0, 2, 1, 3, 4});

        video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

        tensor2video(video, batchSize, 1, 3, height, width, "D:\\test\\video\\test_");
	}
	
	public static void video_dit_train_i2v() throws Exception {
		
		String dataPath = "D:\\dataset\\video\\1w_latend.bin";
		String imgDataPath = "D:\\dataset\\video\\1W_img_latend.bin";
		String clipDataPath = "D:\\dataset\\video\\1w_clip.bin";
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        
        int batchSize = 5;
        int latendDim = 128;
        int numFrames = 3;
        int height = 11;
        int width = 20;
        int textEmbedDim = 768;
        int maxContext = 77;
        
        LatendDatasetI2V dataLoader = new LatendDatasetI2V(dataPath, imgDataPath, clipDataPath, batchSize, latendDim, numFrames, height, width, maxContext, textEmbedDim, BinDataType.float32);
		
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int patchSize = 1;
        int hiddenSize = 1152;

        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;
        
        OmegaVideoI2V dit = new OmegaVideoI2V(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, height, width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, token_drop, path_drop_prob);
    	dit.CUDNN = true;
        dit.learnRate = 1e-4f;
         
        String model_path = "D:\\dataset\\video\\models\\video_dit_b_12.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        
        loadMS(meanStdPath, mean, std);

        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 60, 0.00001f, batchSize, LearnRateUpdate.NONE, false);
        optimizer.train_video_dit_ICPlan(dataLoader, icplan, "D:\\dataset\\video\\models\\video_dit_b_", mean, std, 2);
        
        String save_model_path = "D:\\dataset\\video\\models\\video_dit_b.model";
        ModelUtils.saveModel(dit, save_model_path);
	}
	
	public static void video_dit_test_i2v() throws Exception {
		int num_frames = 17;
		int height = 352;
		int width = 640;
		int patch_size_t = 1;
		int patch_size = 4;
        int vae_numFrames = 3;
        int vae_height = 11;
        int vae_width = 20;
		int[] block_out_channels = new int[] {128, 256, 512, 512};
		int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
		boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
		
		LTXVideo_VAE_Encoder vae_encoder = new LTXVideo_VAE_Encoder(LossType.MSE, UpdaterType.adamw, 1, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		LTXVideo_VAE_Decoder vae_decoder = new LTXVideo_VAE_Decoder(LossType.MSE, UpdaterType.adamw, vae_numFrames, vae_height, vae_width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
		vae_encoder.CUDNN = true;
		vae_decoder.CUDNN = true;
		String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
        ModelUtils.loadModel(vae_encoder, vae_decoder, save_model_path);
        
        int latendDim = 128;
        int ditHeadNum = 16;
        int depth = 16;
        int timeSteps = 1000;
        int patchSize = 1;
        int hiddenSize = 1152;

        float token_drop = 0.0f;
        float path_drop_prob = 0.05f;

        OmegaVideoI2V dit = new OmegaVideoI2V(LossType.MSE, UpdaterType.adamw, latendDim, vae_numFrames, vae_height, vae_width, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, token_drop, path_drop_prob);
    	dit.CUDNN = true;
        dit.learnRate = 2e-4f;
        
        ICPlan icplan = new ICPlan(dit.tensorOP);
        
        String model_path = "D:\\dataset\\video\\models\\video_dit_b_4.model";
        ModelUtils.loadModel(dit, model_path);
        
        Tensor mean = new Tensor(latendDim, 1, 1, 1, true);
        Tensor std = new Tensor(latendDim, 1, 1, 1, true);
        String meanStdPath = "D:\\dataset\\video\\1w_mean_std.bin";
        loadMS(meanStdPath, mean, std);
        
        int batchSize = 2;
        
        int thw = vae_numFrames * vae_height * vae_width;
        
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        
        Tensor noise = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor latend = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        Tensor eps = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor noise2 = new Tensor(batchSize, dit.inChannel * dit.num_frames, dit.height, dit.width, true);
        
        Tensor video = new Tensor(batchSize * num_frames, 3, height, width, true);
        
        Tensor[][] cs = RoPE3DKernel.init3DRoPE(vae_numFrames, vae_height, vae_width, hiddenSize, ditHeadNum, 1f, 1.4375f, 2.5f);
        Tensor[] cos = cs[0];
        Tensor[] sin = cs[1];

        dit.RUN_MODEL = RunModel.EVAL;

        String[] imgPaths = new String[] {
        		"C:\\Users\\Administrator\\Desktop\\test_video\\ScreenShot_2026-05-21_165922_765.png",
        		"C:\\Users\\Administrator\\Desktop\\test_video\\ScreenShot_2026-05-26_184055_227.png"
        };
        
        Tensor img_input = new Tensor(batchSize, 3, height, width, true);
        
        VideoReaderExample.loadVImg2Tensor(imgPaths[0], 1, height, width, img_input, 0);
        VideoReaderExample.loadVImg2Tensor(imgPaths[1], 1, height, width, img_input, 1);
        
        img_input.hostToDevice();
        
        int[] shape = new int[] {batchSize, latendDim, vae_numFrames, vae_height, vae_width};
        
        Tensor img_latend = vae_encoder.encode(img_input);
        
        icplan.latend_norm(img_latend, mean, std, 1 * vae_height * vae_width);    
        
        for(int i = 0;i<10;i++) {
        	
        	System.out.println("start create test videos.");

            GPUOP.getInstance().cudaRandn(noise);
            noise.copyGPU(noise2);
            
            Tensor sample = icplan.forward_with_path_drop_cfg(dit, shape, noise, img_latend, t, cos, sin, latend, eps, 1.0f);

            icplan.latend_un_norm(sample, mean, std, thw);

            Tensor result = vae_decoder.decode(sample);
//          result.showShape("result");
            vae_decoder.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\"+i);
            
            System.out.println("finish create.");
            
            sample = icplan.forward_with_path_drop_cfg(dit, shape, noise2, img_latend, t, cos, sin, latend, eps, 2.0f);
            
            icplan.latend_un_norm(sample, mean, std, thw);

            result = vae_decoder.decode(sample);
            vae_decoder.tensorOP.permute(result, video, new int[] {batchSize, 3, num_frames, height, width}, new int[] {batchSize, num_frames, 3, height, width}, new int[] {0, 2, 1, 3, 4});

            video.data = MatrixOperation.clampSelf(video.syncHost(), -1, 1);

            tensor2video(video, batchSize, num_frames, 3, height, width, "D:\\test\\video\\t_"+i);
            
            System.out.println("finish create.");
        }
        
	}
	
	public static void loadMS(String meanStdPath, Tensor mean, Tensor std) {
        try (RandomAccessFile File = new RandomAccessFile(meanStdPath, "r")) {
        	com.omega.engine.nn.network.utils.ModelUtils.loadParams(File, mean);
        	com.omega.engine.nn.network.utils.ModelUtils.loadParams(File, std);
            System.out.println("load mean std success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
	}
	
	public static void tensor2video(Tensor org, int N, int F, int C, int H, int W, String outputPath) throws Exception {
		ImageUtils utils = new ImageUtils();
        for (int b = 0; b < N; b++) {
        	List<BufferedImage> frames = new ArrayList<BufferedImage>();
        	for(int f = 0;f<F;f++) {
        		int bf = b * F + f;
        		float[] once = org.getByNumber(bf);
        		int[][] rgb = ImageUtils.color2argb(once, C, H, W);
//        		System.err.println(JsonUtils.toJson(rgb));
        		BufferedImage bufferedImage = utils.convertRGBImage(rgb);
        		frames.add(bufferedImage);
        	}
        	VideoReaderExample.writeFramesToVideo(frames, outputPath + "_" + b + ".mp4");
        }
	}
	
	public static void main(String[] args) {
		
		try {
			 
//			complate_ms();
			
//			video_dit_train();
			
//			video_dit_test();
			
//			video_dit_train2();
			
			video_dit_test2();
			
//			video_dit_train_i2v();
//			video_dit_test_i2v();
//			test_img();
			
//			test_vae_img();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
