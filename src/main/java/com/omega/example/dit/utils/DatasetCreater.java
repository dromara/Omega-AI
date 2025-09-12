package com.omega.example.dit.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import jcuda.driver.JCudaDriver;

public class DatasetCreater {
	
	
    public static void writeTensor(Tensor x, FileOutputStream writer) throws IOException {
//        System.out.println("writing.");
        if(x.isHasGPU()){
        	x.syncHost();
        }
        for(int i = 0;i<x.dataLength;i++) {
        	float s = x.data[i];
            byte[] bs = ModelUtils.float2byte(s);
            writer.write(bs);
        }
    }
    
    public static void createLatendDataset() {
    	
    	try {
    		
//    		String outputPath = "D:\\dataset\\amine\\dalle_latend.bin";
    		String clipDataPath = "D:\\dataset\\amine\\dalle_full_clip.bin";
    		
        	String labelPath = "D:\\dataset\\labels.json";
            String imgDirPath = "D:\\dataset\\images_256_256\\";
            boolean horizontalFilp = false;
            int imgSize = 256;
            int maxContextLen = 77;
            int batchSize = 256;
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
            
//            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);
            
            for(int it = 0;it<indexs.length;it++) {
            	 dataLoader.loadData(indexs[it], input, label, labels, eosIds);
//            	 Tensor latend = vae.encode(input);
//                 JCudaDriver.cuCtxSynchronize();
//                 writeTensor(latend, writer);
            	 Tensor condInput = clip.get_full_clip_prompt_embeds(label);
                 writeTensor(condInput, clipWriter);
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
    
    public static void createLatendDataset_vavae() {
    	
    	try {
    		
    		String outputPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
    		String clipDataPath = "D:\\dataset\\amine\\dalle_vavae_clip.bin";
    		
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
    	
    	int batchSize = 20;
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

    	String dataPath = "D:\\dataset\\amine\\dalle_vavae_latend.bin";
    	
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
            MBSGDOptimizer.showImgs("D:\\test\\va_vae\\", output, "test", mean, std);
            
            ModelUtils.readFloat(file, latend);
            
            output = vae.decode(latend);
            
            output.showShape();
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            /**
             * print image
             */
            MBSGDOptimizer.showImgs("D:\\test\\va_vae\\", output, "test2", mean, std);
            
            ModelUtils.readFloat(file, latend);
            
            output = vae.decode(latend);
            
            output.showShape();
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            /**
             * print image
             */
            MBSGDOptimizer.showImgs("D:\\test\\va_vae\\", output, "test3", mean, std);
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    	
    }
    
    public static void main(String[] args) {	
		 
        try {

        	createLatendDataset();
        	
//        	createLatendDataset2();
        	
//        	createLatendDataset_vavae();
        	
//        	testLatendData();
        	
//        	test_vavae_latend();
        	
//        	createLatendDataset3();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
	
}
