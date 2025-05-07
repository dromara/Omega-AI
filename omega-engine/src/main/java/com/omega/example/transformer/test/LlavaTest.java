package com.omega.example.transformer.test;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.nn.network.Llava;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.dataset.LVMPreTrainDataset;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import com.omega.example.yolo.data.ImageLoader;

import java.util.Arrays;
import java.util.Scanner;

public class LlavaTest {
    public static ClipVision createClipVision() {
        boolean bias = true;
        int channel = 3;
        int imgSize = 224;
        int patchSize = 32;
        int headNum = 12;
        int nLayers = 12;
        int clip_time = 50;
        int embedDim = 768;
        ClipVision network = new ClipVision(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, channel, imgSize, patchSize, headNum, nLayers, clip_time, embedDim, bias, false);
        network.time = clip_time;
        network.CUDNN = true;
        String clipWeight = "H:\\model\\clip_vision_weights.json";
        ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, false);
        System.out.println("clip vision is ready.");
        return network;
    }

    public static void llava_pertrain() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 2;
            int imageSize = 224;
            int max_len = 512;
            int imageTime = 50;
            int embedDim = 512;
            int visionOutDim = 768;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            String trainPath = "I:\\BaiduNetdiskDownload\\dataset\\LLaVA-Pretrain\\chat-translated.json";
            String trainImagePath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            LVMPreTrainDataset trainData = new LVMPreTrainDataset(trainPath, trainImagePath, "", imageSize, max_len, batchSize, true, tokenizer);
            Llava network = new Llava(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, imageTime, embedDim, visionOutDim, bias, dropout);
            network.learnRate = 1e-4f;
            ClipVision clipVision = createClipVision();
            String model_path = "H:\\model\\llama3-26m-chinese.model";
            ModelUtils.loadPertrainModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.train_llava_chinese(trainData, clipVision, 8, true);
            String save_model_path = "H:\\model\\llava-26m-chinese.model";
            ModelUtils.saveModel(network, save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llava_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 2;
            int imageSize = 224;
            int max_len = 512;
            int imageTime = 50;
            int embedDim = 512;
            int visionOutDim = 768;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            String trainPath = "I:\\BaiduNetdiskDownload\\dataset\\LLaVA-Instruct\\llava_instruct_230k.json";
            String trainImagePath = "I:\\BaiduNetdiskDownload\\dataset\\sft_images\\";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            LVMPreTrainDataset trainData = new LVMPreTrainDataset(trainPath, trainImagePath, "", imageSize, max_len, batchSize, true, tokenizer);
            Llava network = new Llava(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, imageTime, embedDim, visionOutDim, bias, dropout);
            network.learnRate = 1e-4f;
            ClipVision clipVision = createClipVision();
            String model_path = "H:\\model\\llava-26m-chinese.model";
            ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 5, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.train_llava_chinese(trainData, clipVision, 8, false);
            String save_model_path = "H:\\model\\llava-26m-chinese-sft.model";
            ModelUtils.saveModel(network, save_model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            Tensor testInput = null;
            Tensor imageInput = new Tensor(1, 3, imageSize, imageSize, true);
            String bos = tokenizer.sos_str() + "user\n";
            int start = tokenizer.encodeInt(bos).length;
            Tensor indice = new Tensor(1, 1, 1, 1, new float[]{start}, true);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                System.out.println("请输入图片位置:");
                String imagePath = scanner.nextLine();
                ImageLoader.loadImage(imageInput, 0, imagePath, imageSize, imageSize, trainData.mean, trainData.std, true);
                imageInput.hostToDevice();
                clipVision.forward(imageInput);
                input_txt = input_txt.toLowerCase();
                String qaStr = tokenizer.sos_str() + "user\n" + trainData.image_special_token + "\n" + input_txt + tokenizer.eos_str() + "\n";
                //				System.out.println(qaStr);
                int[] idx = tokenizer.encodeInt(qaStr);
                int startLen = idx.length;
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(clipVision.getEncoder().getImageEncoders(), indice, cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDXTopN(output, idx.length - 1, 8, network.cudaManager);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos) {
                        break;
                    }
                    input = loadByTxtToIdx(testInput, idx);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                int[] awIdx = Arrays.copyOfRange(idx, startLen, idx.length);
                System.out.println("chatbot:" + tokenizer.decode(awIdx).replaceAll("<s>assistant\n", ""));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llava_sft_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int imageSize = 224;
            int max_len = 512;
            int imageTime = 50;
            int embedDim = 512;
            int visionOutDim = 768;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            int vocab_size = 6400;
            float[] mean = new float[]{0.48145466f, 0.4578275f, 0.40821073f};
            float[] std = new float[]{0.26862954f, 0.26130258f, 0.27577711f};
            String image_special_token = "<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            Llava network = new Llava(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, imageTime, embedDim, visionOutDim, bias, dropout);
            ClipVision clipVision = createClipVision();
            String model_path = "H:\\model\\llava-26m-chinese-sft3.model";
            ModelUtils.loadModel(network, model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            Tensor testInput = null;
            Tensor imageInput = new Tensor(1, 3, imageSize, imageSize, true);
            String bos = tokenizer.sos_str() + "user\n";
            int start = tokenizer.encodeInt(bos).length;
            Tensor indice = new Tensor(1, 1, 1, 1, new float[]{start}, true);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                System.out.println("请输入图片位置:");
                String imagePath = scanner.nextLine();
                ImageLoader.loadImage(imageInput, 0, imagePath, imageSize, imageSize, mean, std, true);
                imageInput.hostToDevice();
                clipVision.forward(imageInput);
                input_txt = input_txt.toLowerCase();
                String qaStr = tokenizer.sos_str() + "user\n" + image_special_token + "\n" + input_txt + tokenizer.eos_str() + "\n";
                //				System.out.println(qaStr);
                int[] idx = tokenizer.encodeInt(qaStr);
                int startLen = idx.length;
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(clipVision.getEncoder().getImageEncoders(), indice, cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDXTopN(output, idx.length - 1, 3, network.cudaManager);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos) {
                        break;
                    }
                    input = loadByTxtToIdx(testInput, idx);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                int[] awIdx = Arrays.copyOfRange(idx, startLen, idx.length);
                System.out.println("chatbot:" + tokenizer.decode(awIdx).replaceAll("<s>assistant\n", ""));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static int output2NextIDXTopN(Tensor output, int nextTokenIdx, int topK, CUDAManager cudaManager) {
        SoftmaxKernel kernel = new SoftmaxKernel(cudaManager);
        Tensor tmp = new Tensor(1, 1, 1, output.width, true);
        Tensor prof = new Tensor(1, 1, 1, output.width, true);
        if (nextTokenIdx < output.number) {
            tmp.hostToDevice(MatrixOperation.multiplication(output.getByNumber(nextTokenIdx), 0.7f));
            kernel.softmax_out(tmp, prof);
            return pickTopN(prof.syncHost(), topK);
        }
        return 0;
    }

    public static int pickTopN(float[] x, int n) {
        float[] sort = Arrays.copyOf(x, x.length);
        Arrays.sort(sort);
        float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
        float v = topN[RandomUtils.getRandomNumber(topN)];
        for (int i = 0; i < x.length; i++) {
            if (v == x[i]) {
                return i;
            }
        }
        return 0;
    }

    public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs) {
        //		System.out.println(idxs.length);
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            //			llava_pertrain();
            //			llava_sft();
            llava_sft_predict();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

