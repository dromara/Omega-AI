package com.omega.example.opensora.vae.utils;

import cn.hutool.setting.dialect.Props;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.dataset.LatendDataset;
import com.omega.example.opensora.vae.dataset.VideoDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.transformer.utils.bpe.BinDataType;

import jcuda.driver.JCudaDriver;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

public class DatasetCreator {

    public static void writeTensor(Tensor x, FileOutputStream writer) throws IOException {
        if(x.isHasGPU()){
            x.syncHost();
        }
        for(int i = 0;i<x.dataLength;i++) {
            float s = x.data[i];
            byte[] bs = ModelUtils.float2byte(s);
            writer.write(bs);
        }
    }

    /**
     * 视频处理
     * latent的二进制文件
     *
     * @param config 配置此参数， 可从指定文件读取
     * @param mode test or create
     * @throws Exception 外部需捕获处理异常
     */
    public static void latentDataset(Props config, String mode) throws Exception{
        // 视频图片尺寸
        int imgSize = config.getInt("img_size", 256);
        // 提取帧数
        int num_frames = config.getInt("num_frames", 9);;

        // 数据集路径 csv 文件
        String dataPath = config.getStr("data_path");
        // 视频提取图片后的保存路径, 保存规则为 img_dir_path/video_name/{frame_number}.png
        String imgDirPath = config.getStr("img_dir_path");
        int maxContextLen = config.getInt("max_context_len", 77);
        int batchSize = config.getInt("batch_size", 2);;

        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        // 分词器
        String vocabPath = config.getStr("vocab_path");
        String mergesPath = config.getStr("merges_path");
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, config.getInt("sos", 49406), config.getInt("eos", 49407));

        // clip参数
        int maxPositionEmbeddingsSize = config.getInt("clip_max_position_embedding_size", 77);
        int vocabSize = config.getInt("clip_vocab_size", 49408);
        int headNum = config.getInt("clip_head_num", 12);
        int n_layers = config.getInt("clip_n_layers", 12);
        int textEmbedDim = config.getInt("clip_text_embed_dim", 768);
        int intermediateSize = config.getInt("clip_intermediate_size", 3072);
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = config.getStr("clip_model_path");
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        // 数据集加载
        VideoDataLoaderEN dataLoader = new VideoDataLoaderEN(bpe, dataPath, imgDirPath, ".png", num_frames, imgSize, imgSize, maxContextLen, batchSize, mean, std);

        // WFVAE 参数
        int latendDim = config.getInt("latend_dim", 8);
        int base_channels = config.getInt("base_channels", 128);
        int en_energy_flow_hidden_size = config.getInt("en_energy_flow_hidden_size", 64);
        int de_energy_flow_hidden_size = config.getInt("de_energy_flow_hidden_size", 128);
        int num_res_blocks = config.getInt("num_res_blocks", 2);
        int connect_res_layer_num =  config.getInt("connect_res_layer_num", 2);

        // 实例化 WFVAE
        WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        network.init();

        // 加载WFVAE权重
        String vaeWeight = config.getStr("wfvae_weight_path");
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);

        // latent bin 文件路径
        String outputDataPath = config.getStr("output_data_path");
        // clip 结果 bin 文件路径
        String clipDataPath = config.getStr("clip_data_path");

        if (mode.equals("create")) {
            // vae输出保存路径
            File file = new File(outputDataPath);
            FileOutputStream writer = new FileOutputStream(file);

            // clip向量保存路径
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);

            // vae 输入
            Tensor org_input = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            // vae 输入需转置
            Tensor input = new Tensor(batchSize, network.getChannel() * num_frames, imgSize, imgSize, true);

            Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
            String[] labels = new String[batchSize];
            Tensor eosIdx = new Tensor(batchSize, 1, 1, 1, true);
            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);

            int[][] indexs = dataLoader.order();
            for (int it = 0; it < indexs.length; it++) {
                dataLoader.loadData(indexs[it], org_input, label, labels, eosIdx);
                network.tensorOP.permute(org_input, input, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
                Tensor latend = network.encode(input);
                JCudaDriver.cuCtxSynchronize();
                writeTensor(latend, writer);
                clip.get_clip_prompt_embeds(label, eosIdx, condInput);
                writeTensor(condInput, clipWriter);
                System.out.println(it + "/" + indexs.length + " finish.");
            }
        } else if (mode.equals("test")) {
            Tensor output = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            Tensor latend = new Tensor(batchSize, 24, 32, 32, true);
            RandomAccessFile file = new RandomAccessFile(outputDataPath, "r");

            ModelUtils.readFloat(file, latend);
            Tensor vaeOutput = network.decode(latend);
            network.tensorOP.permute(vaeOutput, output, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            MBSGDOptimizer.showVideos(config.getStr("test_path"), num_frames, output, 0+"", mean, std);
        } else if (mode.equals("dataset")) {

            LatendDataset dataLoaderLantend = new LatendDataset(outputDataPath, clipDataPath, batchSize, 24, 32, 32, 1, textEmbedDim, BinDataType.float32);
            Tensor latend = new Tensor(batchSize, dataLoaderLantend.channel, dataLoaderLantend.height, dataLoaderLantend.width, true);
            Tensor condInput = new Tensor(batchSize, 1, 1, dataLoaderLantend.clipEmbd, true);

            dataLoaderLantend.loadData(new int[]{0,1}, latend, condInput, 0);
            dataLoaderLantend.loadData(new int[]{0,1}, latend, condInput, 0);

            Tensor output = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            Tensor vaeOutput = network.decode(latend);
            network.tensorOP.permute(vaeOutput, output, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            MBSGDOptimizer.showVideos(config.getStr("test_path"), num_frames, output, 0+"", mean, std);
        }
    }
    
    public static void latentDataset2(Props config, String mode) throws Exception{
        // 视频图片尺寸
        int imgSize = config.getInt("img_size", 256);
        // 提取帧数
        int num_frames = config.getInt("num_frames", 9);

        // 数据集路径 csv 文件
        String dataPath = "D:\\dataset\\pexels_45k\\train_set.csv";
        // 视频提取图片后的保存路径, 保存规则为 img_dir_path/video_name/{frame_number}.png
        String imgDirPath = "D:\\dataset\\t2v_dataset\\";
        int maxContextLen = config.getInt("max_context_len", 77);
        int batchSize = config.getInt("batch_size", 2);;

        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        // 分词器
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, config.getInt("sos", 49406), config.getInt("eos", 49407));

        // clip参数
        int maxPositionEmbeddingsSize = config.getInt("clip_max_position_embedding_size", 77);
        int vocabSize = config.getInt("clip_vocab_size", 49408);
        int headNum = config.getInt("clip_head_num", 12);
        int n_layers = config.getInt("clip_n_layers", 12);
        int textEmbedDim = config.getInt("clip_text_embed_dim", 768);
        int intermediateSize = config.getInt("clip_intermediate_size", 3072);
        ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContextLen, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

        // 数据集加载
        VideoDataLoaderEN dataLoader = new VideoDataLoaderEN(bpe, dataPath, imgDirPath, ".png", num_frames, imgSize, imgSize, maxContextLen, batchSize, mean, std);

        // WFVAE 参数
        int latendDim = config.getInt("latend_dim", 8);
        int base_channels = config.getInt("base_channels", 128);
        int en_energy_flow_hidden_size = config.getInt("en_energy_flow_hidden_size", 64);
        int de_energy_flow_hidden_size = config.getInt("de_energy_flow_hidden_size", 128);
        int num_res_blocks = config.getInt("num_res_blocks", 2);
        int connect_res_layer_num =  config.getInt("connect_res_layer_num", 2);

        // 实例化 WFVAE
        WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
        network.CUDNN = true;
        network.learnRate = 0.0001f;
        network.init();

        // 加载WFVAE权重
        String vaeWeight = "D:\\models\\wfvae-s.json";
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);

        String outputDataPath = "D:\\dataset\\wfvae\\video_latend.bin";
		String clipDataPath = "D:\\dataset\\wfvae\\video_clip.bin";

        if (mode.equals("create")) {
            // vae输出保存路径
            File file = new File(outputDataPath);
            FileOutputStream writer = new FileOutputStream(file);

            // clip向量保存路径
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);

            // vae 输入
            Tensor org_input = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            // vae 输入需转置
            Tensor input = new Tensor(batchSize, network.getChannel() * num_frames, imgSize, imgSize, true);

            Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
            String[] labels = new String[batchSize];
            Tensor eosIdx = new Tensor(batchSize, 1, 1, 1, true);
            Tensor condInput = new Tensor(batchSize, 1, 1, textEmbedDim, true);

            int[][] indexs = dataLoader.order();
            for (int it = 0; it < indexs.length; it++) {
                dataLoader.loadData(indexs[it], org_input, label, labels, eosIdx);
                network.tensorOP.permute(org_input, input, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
                Tensor latend = network.encode(input);
                JCudaDriver.cuCtxSynchronize();
                writeTensor(latend, writer);
                clip.get_clip_prompt_embeds(label, eosIdx, condInput);
                writeTensor(condInput, clipWriter);
                System.out.println(it + "/" + indexs.length + " finish.");
            }
        } else if (mode.equals("test")) {
            Tensor output = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            Tensor latend = new Tensor(batchSize, 24, 32, 32, true);
            RandomAccessFile file = new RandomAccessFile(outputDataPath, "r");

            ModelUtils.readFloat(file, latend);
            Tensor vaeOutput = network.decode(latend);
            network.tensorOP.permute(vaeOutput, output, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            MBSGDOptimizer.showVideos(config.getStr("test_path"), num_frames, output, 0+"", mean, std);
        } else if (mode.equals("dataset")) {

            LatendDataset dataLoaderLantend = new LatendDataset(outputDataPath, clipDataPath, batchSize, 40, 32, 32, 1, textEmbedDim, BinDataType.float32);
            Tensor latend = new Tensor(batchSize, dataLoaderLantend.channel, dataLoaderLantend.height, dataLoaderLantend.width, true);
            Tensor condInput = new Tensor(batchSize, 1, 1, dataLoaderLantend.clipEmbd, true);

            dataLoaderLantend.loadData(new int[]{0,1}, latend, condInput, 0);
            dataLoaderLantend.loadData(new int[]{0,1}, latend, condInput, 0);

            Tensor output = new Tensor(batchSize, num_frames * network.getChannel(), imgSize, imgSize, true);
            Tensor vaeOutput = network.decode(latend);
            network.tensorOP.permute(vaeOutput, output, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
            output.syncHost();
            output.data = MatrixOperation.clampSelf(output.data, -1, 1);
            MBSGDOptimizer.showVideos("D:\\dataset\\wfvae\\test\\", num_frames, output, 0+"", mean, std);
        }
    }

    /**
     *
     * @param args 0 "create", "test", "dataset"
     *             1 配置文件路径 resources/dataset/video/video_data.properties
     */
    public static void main(String[] args) {
        Props props = new Props();
        try {
            latentDataset2(props, "create");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
