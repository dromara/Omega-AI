package com.omega.example.dit.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.active.GeluType;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.T5Encoder;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.nn.network.vae.Flux_VAE;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.driver.JCudaDriver;

public class DatasetCreater {
	
	/**
     * 使用缓冲区批量写入
     *
     * @param outputStream
     * @param data
     * @throws IOException
     */
    public static void writeFloatArray(FileOutputStream writer, float[] data) throws IOException {
        final int BUFFER_SIZE = 16384; // 16KB缓冲区
        byte[] buffer = new byte[BUFFER_SIZE];
        int bufferIndex = 0;

        for (float value : data) {
            int fbit = Float.floatToIntBits(value);

            // 直接写入缓冲区（小端序）
            buffer[bufferIndex++] = (byte) fbit ;
            buffer[bufferIndex++] = (byte) (fbit >> 8);
            buffer[bufferIndex++] = (byte) (fbit >> 16);
            buffer[bufferIndex++] = (byte) (fbit >> 24);

            if (bufferIndex >= BUFFER_SIZE) {
            	writer.write(buffer, 0, bufferIndex);
                bufferIndex = 0;
            }
        }

        if (bufferIndex > 0) {
        	writer.write(buffer, 0, bufferIndex);
        }
    }
	
    public static void writeTensor(Tensor x, FileOutputStream writer) throws IOException {
//        System.out.println("writing.");
        if(x.isHasGPU()){
        	x.syncHost();
        }
        writeFloatArray(writer, x.data);
    }
    
    public static void createLatendDataset() {
    	
    	try {
    		
//    		String outputPath = "D:\\dataset\\amine\\dalle_latend.bin";
    		String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
    		String clipMaskDataPath = "D:\\dataset\\amine\\dalle_clip_mask.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_256_256\\";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 1000;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
            Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
            Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
            String[] labels = new String[batchSize];

//            File file = new File(outputPath);
//            FileOutputStream writer = new FileOutputStream(file);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            File clipMaskFile = new File(clipMaskDataPath);
            FileOutputStream clipMaskWriter = new FileOutputStream(clipMaskFile);
            
//            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
            
            for(int it = 0;it<indexs.length;it++) {
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
//            	 Tensor latend = vae.encode(input);
//                 JCudaDriver.cuCtxSynchronize();
//                 writeTensor(latend, writer);
            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
                 writeTensor(condInput, clipWriter);
                 writeTensor(eosIds, clipMaskWriter);
                 System.out.println(it + "/" + indexs.length + " finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatendDataset2() {
    	
    	try {
    		
    		String outputPath = "/root/gpufree-data/txt2img_2m/txt2img_latend.bin";
    		String clipDataPath = "/root/gpufree-data/txt2img_2m/txt2img_clip.bin";
    		
        	String labelPath = "/root/gpufree-data/txt2img_2m/labels.json";
            String imgDirPath = "/root/gpufree-data/processed_images/";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 50;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
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
            String vaeWeight = "/omega/models/sdxl-vae-fp16-fix.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
            Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
            Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
            String[] labels = new String[batchSize];

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
            
            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 clip.get_clip_prompt_embeds(label, eosIds, condInput);
                 writeTensor(condInput, clipWriter);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatendDataset3() {
    	
    	try {
    		
    		String outputPath = "/root/gpufree-data/txt2img_2m/vavae_2m_latend.bin";
    		String clipDataPath = "/root/gpufree-data/txt2img_2m/vavae_2m_clip.bin";
    		
        	String labelPath = "/root/gpufree-data/txt2img_2m/labels.json";
            String imgDirPath = "/root/gpufree-data/processed_images/";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 50;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
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
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
            Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
            Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
            String[] labels = new String[batchSize];

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
            
            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 clip.get_clip_prompt_embeds(label, eosIds, condInput);
                 writeTensor(condInput, clipWriter);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatendDatasetFullClip() {
    	
    	try {
    		
    		String outputPath = "/root/gpufree-data/txt2img_1m/vavae_1m_latend.bin";
    		String clipDataPath = "/root/gpufree-data/txt2img_1m/vavae_1m_clip.bin";
    		
        	String labelPath = "/root/gpufree-data/txt2img_1m/labels.json";
            String imgDirPath = "/root/gpufree-data/txt2img_1m/processed_images/";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 50;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);
            
            int latendDim = 32;
            int num_res_blocks = 2;
            int[] ch_mult = new int[]{1, 1, 2, 2, 4};
            int ch = 128;
            
            VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
            vae.CUDNN = true;
            vae.learnRate = 0.001f;
            vae.RUN_MODEL = RunModel.EVAL;
            String vaeWeight = "/omega/models/vavae.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
            Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
            Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
            String[] labels = new String[batchSize];

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
            	 JCudaDriver.cuCtxSynchronize();
                 writeTensor(condInput, clipWriter);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatendDataset_vavae() {
    	
    	try {
    		
    		String outputPath = "/root/gpufree-data/txt2img_2m_2/vavae_latend.bin";
    		String clipDataPath = "/root/gpufree-data/txt2img_2m_2/vavae_clip.bin";
    		
        	String labelPath = "/root/gpufree-data/txt2img_2m_2/labels.json";
            String imgDirPath = "/root/gpufree-data/diffusiondb_processed/";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 40;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);
            
            
            int latendDim = 32;
            int num_res_blocks = 2;
            int[] ch_mult = new int[]{1, 1, 2, 2, 4};
            int ch = 128;
            
            VA_VAE vae = new VA_VAE(LossType.MSE, UpdaterType.adamw, latendDim, imgSize, ch_mult, ch, num_res_blocks, true);
            vae.CUDNN = true;
            vae.learnRate = 0.001f;
            vae.RUN_MODEL = RunModel.EVAL;
            String vaeWeight = "/omega/models/vavae.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);
            
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
            Tensor label = new Tensor(batchSize * dataLoader.maxContextLen, 1, 1, 1, true);
            Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
            String[] labels = new String[batchSize];

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            Tensor condInput = null;
            
            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 condInput = clip.get_full_clip_prompt_embeds(label);
                 writeTensor(condInput, clipWriter);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void testLatendData() {
    	
    	int batchSize = 20;
    	int channel = 4;
    	int height = 32;
    	int width = 32;
    	
    	int imgSize = 256;
    	float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};

    	Tensor latend = new Tensor(batchSize, channel, height, width, true);
    	
    	int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 4, 4};
        int ch = 128;
        
        SD_VAE vae = new SD_VAE(LossType.MSE, UpdaterType.adamw, latendDim, num_vq_embeddings, imgSize, ch_mult, ch, num_res_blocks, true);
        vae.CUDNN = true;
        vae.RUN_MODEL = RunModel.EVAL;
        String vaeWeight = "D:\\models\\sdxl-vae-fp16-fix\\sdxl-vae-fp16-fix.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(vaeWeight), vae, true);

    	String dataPath = "D:\\dataset\\amine\\dalle_latend.bin";
    	
        try {
        	RandomAccessFile file = new RandomAccessFile(dataPath, "r");
        	
            int number = (int) (file.length() / latend.getOnceSize() / 4);
        	
            System.err.println("count:"+number);
            
            ModelUtils.readFloat(file, latend);
            
            Tensor output = vae.decode(latend);
            
            output.showShape();
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            /**
             * print image
             */
            MBSGDOptimizer.showImgs("D:\\test\\sd_vae2\\", output, "test", mean, std);
            
            ModelUtils.readFloat(file, latend);
            
            output = vae.decode(latend);
            
            output.showShape();
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            /**
             * print image
             */
            MBSGDOptimizer.showImgs("D:\\test\\sd_vae2\\", output, "test2", mean, std);
            
            ModelUtils.readFloat(file, latend);
            
            output = vae.decode(latend);
            
            output.showShape();
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            /**
             * print image
             */
            MBSGDOptimizer.showImgs("D:\\test\\sd_vae2\\", output, "test3", mean, std);
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void test_vavae_latend() {
    	
    	int batchSize = 10;
    	int channel = 32;
    	int height = 16;
    	int width = 16;
    	
    	int imgSize = 256;
    	float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};

    	Tensor latend = new Tensor(batchSize, channel, height, width, true);
    	
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

    	String dataPath = "D:\\dataset\\flux_train_sampled\\vavae_latend.bin";
    	
        try {
        	RandomAccessFile file = new RandomAccessFile(dataPath, "r");
        	
        	file.seek(10 * latend.getOnceSize() * 4);
        	
            int number = (int) (file.length() / latend.getOnceSize() / 4);
        	
            System.err.println("count:"+number);
            
            for(int i = 0;i<100;i++) {

                ModelUtils.readFloat(file, latend);
                
                Tensor output = vae.decode(latend);
                
                output.showShape();
                output.syncHost();
                output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                /**
                 * print image
                 */
                MBSGDOptimizer.showImgs("D:\\test\\va_vae\\", output, "test_"+i, mean, std);
                
            }
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void loadLabels(BPETokenizerEN tokenizer,List<Map<String, Object>> datas, int[] indexs, Tensor label, String[] labels, Tensor eos_idx, int maxContextLen) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("en").toString();
            labels[i] = text;
            //			System.out.println(text); 
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            float eos_id = 0;
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = tokenizer.eos();
                }
                //获取第一个结束符位置
                if(label.data[i * maxContextLen + j] == tokenizer.eos() && eos_id == 0) {
                	eos_id = j;
                }
            }
            eos_idx.data[i] = eos_id;
        }
        /**
         * copy data to gpu.
         */
        label.hostToDevice();
        eos_idx.hostToDevice();
        //		System.out.println(JsonUtils.toJson(label.data));
    }
    
    public static void loadLabels(BPETokenizerEN tokenizer,List<Map<String, Object>> datas, String key, int[] indexs, Tensor label, int maxContextLen, int batchSize) {
        LabelsLoader.load(tokenizer, datas, key, indexs, batchSize, label, maxContextLen);
        /**
         * copy data to gpu.
         */
        label.hostToDevice();
    }
    
    public static void loadLabels(SentencePieceTokenizer tokenizer,List<Map<String, Object>> datas, String key, int[] indexs, Tensor label, Tensor mask, int maxContextLen, int batchSize) {
    	T5LabelsLoader.load(tokenizer, datas, key, indexs, batchSize, label, mask, maxContextLen);
        /**
         * copy data to gpu.
         */
        label.hostToDevice();
    }
    
    public static void createLatend_vavae() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\flux_train_sampled\\vavae_latend.bin";
    		
        	String labelPath = "D:\\dataset\\flux_train_sampled\\metadata.json";
            String imgDirPath = "D:\\dataset\\flux_train_sampled\\images\\";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 40;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".png", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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

            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatend_vavae_512() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\amine\\dalle_vavae_latend_512.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_512_512\\selected_images\\";
            boolean horizontalFilp = false;
            int imgSize = 512;
            int maxContextLen = 77;
            int batchSize = 10;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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

            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatend_flux_vae() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\amine\\dalle_fluxvae_latend.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_256_256\\";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 40;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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

            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createLatend_fluxvae_512() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\amine\\dalle_fluxvae_latend_512.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_512_512\\selected_images\\";
            boolean horizontalFilp = false;
            int imgSize = 512;
            int maxContextLen = 77;
            int batchSize = 10;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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

            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input);
            	 Tensor latend = vae.encode(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(latend, writer);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void test_fluxvae_latend() {
    	
    	int batchSize = 10;
    	int channel = 16;
    	int height = 32;
    	int width = 32;
    	
    	int imgSize = 256;
    	float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};

    	Tensor latend = new Tensor(batchSize, channel, height, width, true);
    	
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

    	String dataPath = "D:\\dataset\\amine\\dalle_fluxvae_latend_512.bin";
    	
        try {
        	RandomAccessFile file = new RandomAccessFile(dataPath, "r");
        	
        	file.seek(10 * latend.getOnceSize() * 4);
        	
            int number = (int) (file.length() / latend.getOnceSize() / 4);
        	
            System.err.println("count:"+number);
            
            for(int i = 0;i<100;i++) {

                ModelUtils.readFloat(file, latend);
                
                Tensor output = vae.decode(latend);
                
                output.showShape();
                output.syncHost();
                output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                /**
                 * print image
                 */
                MBSGDOptimizer.showImgs("D:\\test\\fluxvae\\", output, "test_"+i, mean, std);
                
            }
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void createLatend_vavae_unsample() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\amine\\vavae_latend_m_s.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_256_256\\";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 40;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

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
            
            int[][] indexs = dataLoader.order();
            
            Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

            for(int it = 0;it<indexs.length;it++) {
            	 long start = System.nanoTime();
            	 dataLoader.loadData(indexs[it], input);
            	 Tensor meanStd = vae.encode_unsample(input);
                 JCudaDriver.cuCtxSynchronize();
                 writeTensor(meanStd, writer);
                 System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
            System.out.println("create ["+dataLoader.count+"] finish.");
           
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

    }
    
    public static void createClipData() {
    	
    	int batchSize = 2000;
    	int maxContextLen = 77;
    	
//    	String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
//		String clipMaskDataPath = "D:\\dataset\\amine\\dalle_clip_mask.bin";
    	
//		String labelPath = "/root/gpufree-data/txt2img_2m/labels.json";
//    	String labelPath = "D:\\dataset\\labels.json";
		String clipDataPath = "/root/gpufree-data/txt2img_2m_2/vavae_clip.bin";
		
    	String labelPath = "/root/gpufree-data/txt2img_2m_2/labels.json";
    	
    	Tensor label = new Tensor(batchSize * 77, 1, 1, 1, true);
//        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
//        String[] labels = new String[batchSize];
    	
    	try {
    		
//    		String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
//            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
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
//            String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);
            
    		List<Map<String, Object>> datas = LagJsonReader.readJsonDataSamll(labelPath);
            int count = datas.size();
            System.err.println("data count[" + count + "].");

            int[][] indexs = MathUtils.orderInts(count, batchSize);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
//            File clipMaskFile = new File(clipMaskDataPath);
//            FileOutputStream clipMaskWriter = new FileOutputStream(clipMaskFile);
            
            for(int it = 0;it<indexs.length;it++) {
//            	 loadLabels(bpe, datas, indexs[it], label, labels, eosIds, maxContextLen);
           	 	 loadLabels(bpe, datas, "label", indexs[it], label, maxContextLen, batchSize);
            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
            	 JCudaDriver.cuCtxSynchronize();
                 writeTensor(condInput, clipWriter);
//                 writeTensor(eosIds, clipMaskWriter);
                 System.out.println(it + "/" + indexs.length + " finish.");
            }
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void createT5Data() {
    	
    	int batchSize = 100;
    	int maxContextLen = 120;

//		String t5DataPath = "D:\\dataset\\flux_train_sampled\\vavae_t5.bin";
//    	String labelPath = "D:\\dataset\\flux_train_sampled\\metadata.json";
    	
    	String t5DataPath = "D:\\dataset\\amine\\vavae_t5.bin";
    	String labelPath = "D:\\dataset\\labels.json";
    	
    	Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
    	Tensor mask = new Tensor(batchSize, 1, 1, maxContextLen, true);

    	try {

    		String tokenizer_path = "D:\\models\\t5\\spiece.model";
    		SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path);
    		
    		int time = maxContextLen;
    		int voc_size = 250112;
    		int num_layers = 24;
    		int head_num = 32;
    		int embed_size = 2048;
    		int d_ff = 5120;
    		T5Encoder t5 = new T5Encoder(LossType.MSE, UpdaterType.adamw, voc_size, num_layers, head_num, time, embed_size, d_ff, false);
    		t5.CUDNN = true;
    		t5.RUN_MODEL = RunModel.EVAL;
        	
    		String model_path = "D://models//t5//t5_encoder.model";
    		com.omega.example.transformer.utils.ModelUtils.loadModel(t5, model_path);
            
    		List<Map<String, Object>> datas = LagJsonReader.readJsonDataSamll(labelPath);
            int count = datas.size();
            System.err.println("data count[" + count + "].");

            int[][] indexs = MathUtils.orderInts(count, batchSize);
            
            File clipFile = new File(t5DataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);

            for(int it = 0;it<indexs.length;it++) {
           	 	 loadLabels(tokenizer, datas, "en", indexs[it], label, mask, maxContextLen, batchSize);
            	 Tensor condInput = t5.forward(label, mask);
            	 JCudaDriver.cuCtxSynchronize();
                 writeTensor(condInput, clipWriter);
                 System.out.println(it + "/" + indexs.length + " finish.");
            }
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void createTwoClip() {
    	
    	try {

    		int batchSize = 64;
        	int maxContextLen = 77;
    		
    		String clipDataPath = "D:\\dataset\\amine\\vavae_2clip2.bin";
    		
    		String labelPath = "D:\\dataset\\labels.json";
    		
        	Tensor label = new Tensor(batchSize * 77, 1, 1, 1, true);
    		
    		String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
    		
            int time = maxContextLen;
            int maxPositionEmbeddingsSize = 77;
            int vocabSize = 49408;
            int headNum = 20;
            int n_layers = 32;
            int textEmbedDim = 1280;
            int intermediateSize = 5120;
            ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers, GeluType.NONE);
            clip.id = "0";
            clip.CUDNN = true;
            clip.time = maxContextLen;
            clip.RUN_MODEL = RunModel.EVAL;
            String model_path = "D:\\models\\CLIP-ViT-bigG-14\\CLIP-ViT-bigG-14.model";
            com.omega.example.transformer.utils.ModelUtils.loadModel(clip, model_path);

            int headNum2 = 12;
            int n_layers2 = 12;
            int textEmbedDim2 = 768;
            int intermediateSize2 = 3072;
            ClipTextModel clip2 = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum2, time, vocabSize, textEmbedDim2, maxPositionEmbeddingsSize, intermediateSize2, n_layers2, GeluType.FAST);
            clip2.id = "1";
            clip2.CUDNN = true;
            clip2.time = time;
            clip2.RUN_MODEL = RunModel.EVAL;
            String model_path2 = "D:\\models\\clip-vit-large-patch14\\clip-vit-large-patch14.model";
            com.omega.example.transformer.utils.ModelUtils.loadModel(clip2, model_path2);
            
    		List<Map<String, Object>> datas = LagJsonReader.readJsonDataSamll(labelPath);
            int count = datas.size();
            System.err.println("data count[" + count + "].");

            int[][] indexs = MathUtils.orderInts(count, batchSize);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            Tensor condInputCat = new Tensor(batchSize * 77, 1, 1, textEmbedDim2 + textEmbedDim, true);
            
            for(int it = 0;it<indexs.length;it++) {
            	 long start  = System.nanoTime();
            	 loadLabels(bpe, datas, "label", indexs[it], label, maxContextLen, batchSize);
            	 System.out.println((System.nanoTime() - start)/1e6);
            	 long start1  = System.nanoTime();
            	 Tensor condInput2 = clip2.get_full_clip_prompt_embeds(label);
            	 System.out.println((System.nanoTime() - start1)/1e6);
            	 long start2  = System.nanoTime();
            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
            	 System.out.println((System.nanoTime() - start2)/1e6);
            	 clip.tensorOP.cat_width(condInput2, condInput, condInputCat, condInput2.width, condInput.width);
            	 
            	 JCudaDriver.cuCtxSynchronize();
//                 writeTensor(condInputCat, clipWriter);
                 System.out.println(it + "/" + indexs.length + "["+(System.nanoTime() - start)/1e6+"ms] finish.");
            }
            
    	}catch (Exception e) {
			// TODO: handle exception
    		e.printStackTrace();
		}	
    	
    }
    
    
	public static void create_dinov2() throws Exception {
		
		String outputPath = "D:\\dataset\\dinov2_z.bin";
		
        String labelPath = "D:\\dataset\\labels.json";
		String imgDirPath = "D:\\dataset\\images_224_224\\";
        boolean horizontalFilp = false;
        int imgSize = 224;
        int maxContextLen = 77;
        int batchSize = 60;
        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, ".jpg", imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
		
        int patchSize = 14;
        int hiddenSize = 768;
        int headNum = 12;
        int depth = 12;
        int mlpRatio = 4;
        Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, patchSize, hiddenSize, headNum, depth, mlpRatio);
        dinov.CUDNN = true;
        dinov.RUN_MODEL = RunModel.EVAL;

        String model_path = "D:\\models\\dionv2-14-b.model";
        com.omega.example.transformer.utils.ModelUtils.loadModel(dinov, model_path);
        
        int[][] indexs = dataLoader.order();
        
        Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);

        File file = new File(outputPath);
        FileOutputStream writer = new FileOutputStream(file);

        for(int it = 0;it<indexs.length;it++) {
        	 long start = System.nanoTime();
        	 dataLoader.loadData(indexs[it], input);
        	 Tensor latend = dinov.forward_features(input);
             JCudaDriver.cuCtxSynchronize();
             writeTensor(latend, writer);
             System.out.println(it + "/" + indexs.length + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
        }
        
        System.out.println("create ["+dataLoader.count+"] finish.");
        
    }
    
//    public static void createClipData() {
//    	
//    	int batchSize = 1000;
//    	int maxContextLen = 77;
//    	
//    	
//    	String clipDataPath = "/root/gpufree-data/dalle_full_clip.bin";
//		String clipMaskDataPath = "/root/gpufree-data/dalle_clip_mask.bin";
//    	
//    	String labelPath = "/omega/dataset/labels.json";
//    	
//    	Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
//        Tensor eosIds = new Tensor(batchSize, 1, 1, 1, true);
//        String[] labels = new String[batchSize];
//    	
//    	try {
//    		
//    		String vocabPath = "/omega/models/CLIP-GmP-ViT-L-14/vocab.json";
//            String mergesPath = "/omega/models/CLIP-GmP-ViT-L-14/merges.txt";
//            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
//    		
//            int maxPositionEmbeddingsSize = 77;
//            int vocabSize = 49408;
//            int headNum = 12;
//            int n_layers = 12;
//            int textEmbedDim = 768;
//            int intermediateSize = 3072;
//            ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
//            clip.CUDNN = true;
//            clip.time = maxContextLen;
//            clip.RUN_MODEL = RunModel.EVAL;
//            String clipWeight = "/omega/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
//            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);
//            
//    		List<Map<String, Object>> datas = LagJsonReader.readJsonDataSamll(labelPath);
//            int count = datas.size();
//            System.err.println("data count[" + count + "].");
//
//            int[][] indexs = MathUtils.orderInts(count, batchSize);
//            
//            File clipFile = new File(clipDataPath);
//            FileOutputStream clipWriter = new FileOutputStream(clipFile);
//            
//            File clipMaskFile = new File(clipMaskDataPath);
//            FileOutputStream clipMaskWriter = new FileOutputStream(clipMaskFile);
//            
//            for(int it = 0;it<indexs.length;it++) {
//            	 loadLabels(bpe, datas, indexs[it], label, labels, eosIds, maxContextLen);
//            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
//            	 JCudaDriver.cuCtxSynchronize();
//                 writeTensor(condInput, clipWriter);
//                 writeTensor(eosIds, clipMaskWriter);
//                 System.out.println(it + "/" + indexs.length + " finish.");
//            }
//            
//        } catch (Exception e) {
//            // TODO: handle exception
//            e.printStackTrace();
//        }
//    	
//    }
    
    public static void main(String[] args) {	
		 
        try {

//        	createLatendDataset();
        	
//        	createLatendDataset2();
        	
//        	createLatendDataset_vavae();
        	
//        	testLatendData();
        	
//        	createLatend_vavae_512();
        	
//        	test_vavae_latend();
        	
//        	createLatendDataset3();
        	
//        	createLatend_vavae();
        	
//        	createLatend_vavae_unsample();
        	
//        	createClipData();
        	
//        	createLatendDatasetFullClip();
        	
//        	createTwoClip();
        	
//        	create_dinov2();
        	
//        	createLatend_flux_vae();
        	
//        	createLatend_fluxvae_512();
        	
//        	test_fluxvae_latend();
        	
        	createT5Data();
        	
//        	createLatend_vavae();
        	
//        	test_vavae_latend();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
	
}
