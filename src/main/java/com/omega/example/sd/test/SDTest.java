package com.omega.example.sd.test;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.*;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.tokenizer.bertTokenizer.BertTokenizer;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

import java.util.Scanner;

/**
 * stable diffusion
 *
 * @author Administrator
 */
public class SDTest {
    public static void test_vqvae() {
        int batchSize = 2;
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imageSize, imageSize, 64, batchSize, true, mean, std);
        TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        network.CUDNN = true;
        network.learnRate = 0.001f;
        network.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\vqvae2_256_32_500.model";
        ModelUtils.loadModel(network, vqvae_model_path);
        int[] indexs = new int[]{0, 1, 2, 3};
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        //		Tensor out = network.forward(input);
        Tensor latent = network.encode(input);
        Tensor a = new Tensor(batchSize, 1, 1, 1, true);
        Tensor b = new Tensor(batchSize, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, latendDim, 32, 32, true);
        float beta_1 = 1e-4f;
        float beta_T = 0.02f;
        int T = 1000;
        float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
        float[] alphas = MatrixOperation.subtraction(1, betas);
        float[] alphas_bar = MatrixUtils.cumprod(alphas);
        float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
        float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
        int[] t_data = new int[]{0, 400};
        float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
        float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
        a.setData(exsa1);
        b.setData(exsa2);
        RandomUtils.gaussianRandom(noise, 0, 1);
        dataLoader.addNoise(a, b, latent, noise, network.cudaManager);
        //		latent.showShape();
        //		latent.showDM();
        Tensor out = network.decode(latent);
        out.showShape();
        //		out.showDM();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test", mean, std);
        dataLoader.unNoise(a, b, latent, noise, network.cudaManager);
        out = network.decode(latent);
        out.showShape();
        //		out.showDM();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test_un", mean, std);
        indexs = new int[]{4, 5, 6, 7};
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        //		Tensor out = network.forward(input);
        for (int i = 0; i < 10; i++) {
            long start = System.nanoTime();
            latent = network.encode(input);
            out = network.decode(latent);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start) / 1e6 + "ms.");
        }
        out.showShape();
        //		out.showDM();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test1", mean, std);
    }

    public static void test_vqvae32() {
        int batchSize = 2;
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
        TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        network.CUDNN = true;
        network.learnRate = 0.001f;
        network.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\vqvae2_32_512.model";
        ModelUtils.loadModel(network, vqvae_model_path);
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        SDImageDataLoader dataLoader2 = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int[] indexs = new int[]{0, 1};
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
        Tensor mask = new Tensor(batchSize, 1, 1, maxContextLen, true);
        dataLoader2.loadLabel(indexs, label, mask);
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        //		Tensor out = network.forward(input);
        Tensor latent = network.encode(input);
        latent.showShape();
        //		latent.showDM();
        Tensor out = network.decode(latent);
        out.showShape();
        //		out.showDM();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test", mean, std);
        indexs = new int[]{4, 5, 6, 7};
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        //		Tensor out = network.forward(input);
        Tensor clipOutput = null;
        for (int i = 0; i < 10; i++) {
            long start = System.nanoTime();
            latent = network.encode(input);
            clipOutput = clip.forward(label, mask);
            out = network.decode(latent);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start) / 1e6 + "ms.");
        }
        out.showShape();
        clipOutput.showDM();
        //		out.showDM();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test1", mean, std);
    }

    public static void test_vqvae32_anime() {
        int batchSize = 3;
        int imageSize = 256;
        int z_dims = 128;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] ch_mult = new int[]{1, 2, 2, 4};
        int ch = 128;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, true, mean, std);
        VQVAE2 network = new VQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, ch_mult, ch, num_res_blocks);
        network.CUDNN = true;
        network.learnRate = 0.001f;
        network.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\anime_vqvae2_256.model";
        ModelUtils.loadModel(network, vqvae_model_path);
        String labelPath = "I:\\dataset\\sd-anime\\anime_op\\data.json";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader2 = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        //		int time = maxContextLen;
        //		int maxPositionEmbeddingsSize = 77;
        //		int vocabSize = 49408;
        //		int headNum = 8;
        //		int n_layers = 12;
        //		int textEmbedDim = 512;
        //
        //		ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, n_layers);
        //		clip.CUDNN = true;
        //		clip.time = time;
        //		clip.RUN_MODEL = RunModel.EVAL;
        //
        //		String clipWeight = "H:\\model\\clip-vit-base-patch32.json";
        //		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int[] indexs = new int[]{3960, 3801, 7975};
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
        dataLoader2.loadLabel(indexs, label);
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        dataLoader.loadData(indexs, input);
        JCudaDriver.cuCtxSynchronize();
        Tensor latent = network.encode(input);
        //		latent.showDM("latent");
        latent.showShape();
        //		Tensor latent = new Tensor(batchSize, 4, 32, 32, RandomUtils.gaussianRandom(batchSize * 4 * 32 * 32, 1.0f, 0.0f), true);
        Tensor out = network.decode(latent);
        //		out.showDM("out");
        out.showShape();
        out.syncHost();
        out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        /**
         * print image
         *
         */
        MBSGDOptimizer.showImgs("H:\\vae_dataset\\anime_test256\\", out, "test", mean, std);
        //		indexs = new int[] {4, 5, 6, 7};
        //
        //		dataLoader.loadData(indexs, input);
        //
        //		JCudaDriver.cuCtxSynchronize();
        //
        //		Tensor clipOutput = null;
        //
        //		for(int i = 0;i<10;i++) {
        //			long start = System.nanoTime();
        //			latent = network.encode(input);
        ////			clipOutput = clip.forward(label);
        //			out = network.decode(latent);
        //			JCuda.cudaDeviceSynchronize();
        //			System.err.println((System.nanoTime() - start)/1e6+"ms.");
        ////			clipOutput.showDMByOffsetRed(0, 100, "clipOutput");
        //		}
        //
        //		out.showShape();
        //
        //		out.syncHost();
        //		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
        //
        //		/**
        //		 * print image
        //		 */
        //		MBSGDOptimizer.showImgs("H:\\vae_dataset\\anime_test256\\", out, "test1", mean, std);
    }

    public static void getVQVAE32_scale_factor() {
        int batchSize = 8;
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
        TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        network.CUDNN = true;
        network.learnRate = 0.001f;
        network.RUN_MODEL = RunModel.EVAL;
        String vqvae_model_path = "H:\\model\\vqvae2_256_32_500.model";
        ModelUtils.loadModel(network, vqvae_model_path);
        Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
        int[][] indexs = dataLoader.order();
        Tensor out = new Tensor(batchSize, latendDim, z_dims, z_dims, true);
        for (int i = 0; i < indexs.length; i++) {
            System.err.println(i);
            dataLoader.loadData(indexs[i], input);
            JCudaDriver.cuCtxSynchronize();
            Tensor latent = network.encode(input);
            network.tensorOP.add(out, latent, out);
        }
        System.err.println("finis sum.");
        Tensor sum = new Tensor(1, 1, 1, 1, true);
        network.tensorOP.sum(out, sum, 0);
        float meanV = sum.syncHost()[0] / out.dataLength / indexs.length;
        System.err.println(meanV);
        Tensor onceSum = new Tensor(1, 1, 1, 1, true);
        float sum_cpu = 0.0f;
        for (int i = 0; i < indexs.length; i++) {
            System.err.println(i);
            dataLoader.loadData(indexs[i], input);
            JCudaDriver.cuCtxSynchronize();
            Tensor latent = network.encode(input);
            network.tensorOP.sub(latent, meanV, latent);
            network.tensorOP.pow(latent, 2, latent);
            network.tensorOP.sum(latent, onceSum, 0);
            sum_cpu += onceSum.syncHost()[0];
        }
        System.err.println(sum_cpu);
        System.err.println(sum.syncHost()[0]);
        System.err.println(sum_cpu / sum.syncHost()[0]);
        double scale_factor = Math.sqrt(sum_cpu / out.dataLength / indexs.length);
        System.err.println("scale_factor:" + 1 / scale_factor);
    }

    public static void test_clip() {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int[] indexs = new int[]{0, 1, 2, 3};
        Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
        Tensor mask = new Tensor(batchSize, 1, 1, maxContextLen, true);
        dataLoader.loadLabel(indexs, label, mask);
        Tensor output = null;
        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            output = clip.forward(label, mask);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start) / 1e6 + "ms.");
            output.showShape();
            //			output.showDM();
        }
        output.showDM();
    }

    public static void test_clip_text() {
        int maxContextLen = 77;
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
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
        String clipWeight = "H:\\model\\clip-vit-base-patch32.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        Tensor label = new Tensor(maxContextLen, 1, 1, 1, true);
        String[] txts = new String[]{"sharp focus on the cats eyes.", "cinematic bokeh: ironcat, sharp focus on the cat's eyes", "sharp focus on the cats eyes.", "cinematic bokeh: ironcat, sharp focus on the cat's eyes"};
        for (int i = 0; i < 4; i++) {
            String txt = txts[i];
            System.err.println(txt);
            int[] ids = bpe.encodeInt(txt, 77);
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[j] = ids[j];
                } else {
                    label.data[j] = 0;
                }
            }
            label.hostToDevice();
            label.showDM();
            Tensor output = clip.forward(label);
            output.showDM("output");
        }
    }

    public static void sd_train_pokem() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 1;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_32_256_500.model";
        ModelUtils.loadModel(vae, vae_path);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int convOutChannels = 128;
        int unetHeadNum = 8;
        int[] downChannels = new int[]{32, 48, 64};
        int[] midChannels = new int[]{64, 48};
        int numDowns = 1;
        int numMids = 1;
        int numUps = 1;
        boolean[] attns = new boolean[]{true, true};
        boolean[] downSamples = new boolean[]{true, true};
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 64;
        DiffusionUNetCond unet = new DiffusionUNetCond(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, convOutChannels, unetHeadNum, downChannels, midChannels, downSamples, numDowns, numMids, numUps, timeSteps, tEmbDim, textEmbedDim, maxContextLen, true, attns);
        unet.CUDNN = true;
        unet.learnRate = 0.001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
        optimizer.trainSD(dataLoader, vae, clip);
    }

    public static void sd_train_pokem_32() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 2;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_32_512.model";
        ModelUtils.loadModel(vae, vae_path);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int convOutChannels = 128;
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 96, 128, 256};
        int[] midChannels = new int[]{256, 128};
        int numDowns = 1;
        int numMids = 1;
        int numUps = 1;
        boolean[] attns = new boolean[]{true, true, true};
        boolean[] downSamples = new boolean[]{true, true, true};
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        DiffusionUNetCond unet = new DiffusionUNetCond(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, convOutChannels, unetHeadNum, downChannels, midChannels, downSamples, numDowns, numMids, numUps, timeSteps, tEmbDim, textEmbedDim, maxContextLen, true, attns);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{20, 50, 80};
        optimizer.trainSD(dataLoader, vae, clip);
    }

    public static void tiny_sd_train_pokem_32() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 2;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_32_512.model";
        ModelUtils.loadModel(vae, vae_path);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 128, 256, 512};
        int numLayer = 1;
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        int groupNum = 32;
        DiffusionUNetCond2 unet = new DiffusionUNetCond2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, downChannels, unetHeadNum, numLayer, timeSteps, tEmbDim, maxContextLen, textEmbedDim, groupNum);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{20, 50, 80};
        optimizer.trainTinySD(dataLoader, vae, clip);
        //		String save_model_path = "H:\\model\\vqvae2_128_500.model";
        //		ModelUtils.saveModel(unet, save_model_path);
    }

    public static void tiny_sd_test_pokem_32() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        BertTokenizer tokenizer = new BertTokenizer(tokenizerPath, true, true);
        int maxContextLen = 64;
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_32_512.model";
        ModelUtils.loadModel(vae, vae_path);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 128, 256, 512};
        int numLayer = 2;
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        int groupNum = 32;
        int batchSize = 1;
        DiffusionUNetCond2 unet = new DiffusionUNetCond2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, downChannels, unetHeadNum, numLayer, timeSteps, tEmbDim, maxContextLen, textEmbedDim, groupNum);
        unet.RUN_MODEL = RunModel.TEST;
        unet.CUDNN = true;
        unet.number = batchSize;
        String model_path = "H:\\model\\pm_sd_1000.model";
        ModelUtils.loadModel(unet, model_path);
        Scanner scanner = new Scanner(System.in);
        Tensor latent = new Tensor(batchSize, latendDim, latendSize, latendSize, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor label = new Tensor(batchSize * unet.maxContextLen, 1, 1, 1, true);
        Tensor mask = new Tensor(batchSize, 1, 1, unet.maxContextLen, true);
        while (true) {
            System.out.println("请输入中文:");
            String input_txt = scanner.nextLine();
            if (input_txt.equals("exit")) {
                break;
            }
            input_txt = input_txt.toLowerCase();
            loadLabels(input_txt, label, mask, tokenizer, unet.maxContextLen);
            Tensor condInput = clip.forward(label, mask);
            //			condInput.showDM("condInput");
            String[] labels = new String[]{input_txt, input_txt};
            MBSGDOptimizer.testSD(input_txt, latent, t, condInput, unet, vae, labels);
        }
        scanner.close();
    }

    public static void tiny_sd_train_anime_32() throws Exception {
        String labelPath = "I:\\dataset\\sd-anime\\anime_op\\data.json";
        String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 8;
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
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
        String vaeModel = "anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vaeModel);
        int unetHeadNum = 8;
        int[] downChannels = new int[]{128, 256, 512, 768};
        int numLayer = 2;
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        int groupNum = 32;
        DiffusionUNetCond2 unet = new DiffusionUNetCond2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, downChannels, unetHeadNum, numLayer, timeSteps, tEmbDim, maxContextLen, textEmbedDim, groupNum);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
        //		optimizer.lr_step = new int[] {20,50,80};
        optimizer.trainTinySD_Anime(dataLoader, vae, clip);
        String save_model_path = "/omega/models/sd_anime256.model";
        ModelUtils.saveModel(unet, save_model_path);
    }

    public static void testDataset() {
        String labelPath = "I:\\dataset\\sd-anime\\anime_op\\data.json";
        String imgDirPath = "I:\\dataset\\sd-anime\\anime_op\\256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 77;
        int batchSize = 32;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
        SDImageDataLoaderEN dataLoader = new SDImageDataLoaderEN(bpe, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int[][] indexs = dataLoader.order();
        Tensor input = new Tensor(batchSize, 3, dataLoader.img_h, dataLoader.img_w, true);
        Tensor label = new Tensor(batchSize * 77, 1, 1, 1, true);
        Tensor noise = new Tensor(batchSize, 4, 32, 32, true);
        String[] labels = new String[batchSize];
        for (int it = 0; it < indexs.length; it++) {
            dataLoader.loadData(indexs[it], input, label, noise, labels);
            System.err.println(it * batchSize);
        }
    }

    public static void loadLabels(String text, Tensor label, Tensor mask, BertTokenizer tokenizer, int maxContextLen) {
        int[] ids = tokenizer.encode(text);
        int[] ids_n = new int[ids.length + 2];
        System.arraycopy(ids, 0, ids_n, 1, ids.length);
        ids_n[0] = tokenizer.sos;
        ids_n[ids_n.length - 1] = tokenizer.eos;
        for (int j = 0; j < maxContextLen; j++) {
            if (j < ids_n.length) {
                label.data[j] = ids_n[j];
                mask.data[j] = 0;
            } else {
                label.data[j] = 0;
                mask.data[j] = -10000.0f;
            }
        }
        mask.hostToDevice();
        label.hostToDevice();
    }

    public static void loadLabels(String text, Tensor label, BPETokenizerEN tokenizer, int maxContextLen) {
        int[] ids = tokenizer.encodeInt(text, maxContextLen);
        for (int j = 0; j < maxContextLen; j++) {
            if (j < ids.length) {
                label.data[0 * maxContextLen + j] = ids[j];
            } else {
                label.data[0 * maxContextLen + j] = 0;
            }
        }
        label.hostToDevice();
    }

    public static void tiny_ldm_train_pokem_32() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_32_512.model";
        ModelUtils.loadModel(vae, vae_path);
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 128, 256, 512};
        int numLayer = 1;
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        int groupNum = 32;
        DiffusionUNetCond2 unet = new DiffusionUNetCond2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, downChannels, unetHeadNum, numLayer, timeSteps, tEmbDim, 0, 0, groupNum);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{20, 50, 80};
        optimizer.trainTinySD(dataLoader, vae);
    }

    public static void sd_train_pokem_32_uncond() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
        int imageSize = 256;
        int z_dims = 32;
        int latendDim = 4;
        int num_vq_embeddings = 512;
        int num_res_blocks = 2;
        int[] channels = new int[]{32, 64, 128, 256};
        boolean[] attn_resolutions = new boolean[]{false, false, false, false};
        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        String vae_path = "H:\\model\\vqvae2_256_32_500.model";
        ModelUtils.loadModel(vae, vae_path);
        int convOutChannels = 128;
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 96, 128, 256};
        int[] midChannels = new int[]{256, 128};
        int numDowns = 1;
        int numMids = 1;
        int numUps = 1;
        boolean[] attns = new boolean[]{true, true, true};
        boolean[] downSamples = new boolean[]{true, true, true};
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        DiffusionUNetCond unet = new DiffusionUNetCond(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, convOutChannels, unetHeadNum, downChannels, midChannels, downSamples, numDowns, numMids, numUps, timeSteps, tEmbDim, attns);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{20, 50, 80};
        optimizer.trainSD(dataLoader, vae);
    }

    public static void tiny_sd_predict_anime_32() throws Exception {
        int imgSize = 256;
        int maxContextLen = 77;
        String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
        String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN tokenizer = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
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
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
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
        String vaeModel = "H:\\model\\anime_vqvae2_256.model";
        ModelUtils.loadModel(vae, vaeModel);
        int unetHeadNum = 8;
        int[] downChannels = new int[]{64, 128, 256, 512};
        int numLayer = 2;
        int timeSteps = 1000;
        int tEmbDim = 512;
        int latendSize = 32;
        int groupNum = 32;
        int batchSize = 1;
        DiffusionUNetCond2 unet = new DiffusionUNetCond2(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, downChannels, unetHeadNum, numLayer, timeSteps, tEmbDim, maxContextLen, textEmbedDim, groupNum);
        unet.CUDNN = true;
        unet.learnRate = 0.0001f;
        unet.RUN_MODEL = RunModel.TEST;
        String model_path = "H:\\model\\sd_anime256.model";
        ModelUtils.loadModel(unet, model_path);
        Scanner scanner = new Scanner(System.in);
        //		Tensor latent = new Tensor(batchSize, latendDim, latendSize, latendSize, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, true);
        Tensor label = new Tensor(batchSize * unet.maxContextLen, 1, 1, 1, true);
        Tensor input = new Tensor(batchSize, 3, imgSize, imgSize, true);
        Tensor latent = vae.encode(input);
        while (true) {
            System.out.println("请输入英文:");
            String input_txt = scanner.nextLine();
            if (input_txt.equals("exit")) {
                break;
            }
            input_txt = input_txt.toLowerCase();
            loadLabels(input_txt, label, tokenizer, unet.maxContextLen);
            Tensor condInput = clip.forward(label);
            String[] labels = new String[]{input_txt, input_txt};
            MBSGDOptimizer.testSD(input_txt, latent, t, condInput, unet, vae, labels, "H:\\vae_dataset\\anime_test256\\");
        }
        scanner.close();
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            //			sd_train_pokem();
            //			sd_train_pokem_32();
            //			tiny_sd_train_pokem_32();
            //			tiny_sd_test_pokem_32();
            //			tiny_ldm_train_pokem_32();
            //			getVQVAE32_scale_factor();
            //			sd_train_pokem_32_uncond();
            //			test_vqvae();
            //			test_vqvae32();
            //			test_clip();
            //			test_clip_text();
            //			test_vqvae32_anime();
            tiny_sd_predict_anime_32();
            //			test_clip_text();
            //			testDataset();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
