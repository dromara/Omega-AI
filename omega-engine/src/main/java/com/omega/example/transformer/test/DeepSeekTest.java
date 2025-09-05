package com.omega.example.transformer.test;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.parallel.dp.DP;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.dataset.PreTrainDataset2;
import com.omega.example.transformer.dataset.SFTBinDataset;
import com.omega.example.transformer.dataset.parallel.ParallelDataLoader;
import com.omega.example.transformer.dataset.parallel.ThreadDataset;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import com.omega.example.transformer.utils.bpe.BinDataType;

import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DeepSeekTest {
    public static void train_pretrain() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 32;
            int max_len = 512;
            int embedDim = 768;
            int head_num = 8;
            int nKVHeadNum = 2;
            int decoderNum = 8;
            String trainPath = "H:\\H:\\transformer_dataset\\pretrain_hq.jsonl";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            PreTrainDataset2 trainData = new PreTrainDataset2(trainPath, max_len, batchSize, tokenizer);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 5e-4f;
            network.CLIP_GRAD_NORM = true;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 4, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.trainLlama3_chinese(trainData, 8, true, "/omega/models/llama3-26-base-zh");
            String save_model_path = "/omega/models/llama3-26-base-zh.model";
            ModelUtils.saveModel(network, save_model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                System.out.println("user:" + input_txt);
                int[] idx = tokenizer.encodeInt(input_txt);
                int startLen = idx.length;
                Tensor input = trainData.loadByTxtToIdx(idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDX(output, idx.length - 1);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos) {
                        break;
                    }
                    input = trainData.loadByTxtToIdx(idx);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                System.out.println("chatbot:" + tokenizer.decode(idx));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void dp_train_pretrain() {
        try {
            int[] deviceIds = new int[]{0, 1, 2, 3};
            NetworkType networkType = NetworkType.LLAMA3;
            int max_len = 512;
            int embedDim = 512;
            int headNum = 8;
            int nKVHeadNum = 2;
            int decoderNum = 16;
            int vocabSize = 6400;
            int batchSize = 16;
            float lr = 5e-4f;
            String trainPath = "/omega/dataset/pretrain_hq_6400.bin";
            String vocabPath = "/omega/models/vocab.json";
            String mergesPath = "/omega/models/merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTBinDataset trainData = new SFTBinDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            ParallelDataLoader pdl = new ParallelDataLoader(trainData, deviceIds);
            Llama3Parameters parameters = new Llama3Parameters(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, vocabSize, max_len, embedDim, true, false, false, lr);
            DP dp = new DP(deviceIds, 0, networkType, parameters, pdl, 2);
            dp.train();
            String save_model_path = "/omega/models/llama3-26-base-zh.model";
            ModelUtils.saveModel((Llama3) dp.getMaster(), save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void dp_train_full_sft() {
        try {
            int[] deviceIds = new int[]{0, 1, 2, 3};
            NetworkType networkType = NetworkType.LLAMA3;
            int max_len = 512;
            int embedDim = 512;
            int headNum = 8;
            int nKVHeadNum = 2;
            int decoderNum = 16;
            int vocabSize = 6400;
            int batchSize = 24;
            float lr = 5e-4f;
            String trainPath = "/omega/dataset/sft_512_6400.bin";
            String vocabPath = "/omega/models/vocab.json";
            String mergesPath = "/omega/models/merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTBinDataset trainData = new SFTBinDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            ParallelDataLoader pdl = new ParallelDataLoader(trainData, deviceIds);
            Llama3Parameters parameters = new Llama3Parameters(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, vocabSize, max_len, embedDim, false, false, false, lr);
            DP dp = new DP(deviceIds, 0, networkType, parameters, pdl, 6);
            dp.load("/omega/models/llama3-26-base-zh.model");
            dp.train();
            String save_model_path = "/omega/models/llama3-26-fullsft.model";
            ModelUtils.saveModel((Llama3) dp.getMaster(), save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void dp_train_1024_sft() {
        try {
            int[] deviceIds = new int[]{0, 1, 2, 3};
            NetworkType networkType = NetworkType.LLAMA3;
            int max_len = 1024;
            int embedDim = 512;
            int headNum = 8;
            int nKVHeadNum = 2;
            int decoderNum = 16;
            int vocabSize = 6400;
            int batchSize = 8;
            float lr = 5e-4f;
            String trainPath = "/omega/dataset/sft_1024_6400.bin";
            String vocabPath = "/omega/models/vocab.json";
            String mergesPath = "/omega/models/merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTBinDataset trainData = new SFTBinDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            ParallelDataLoader pdl = new ParallelDataLoader(trainData, deviceIds);
            Llama3Parameters parameters = new Llama3Parameters(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, vocabSize, max_len, embedDim, false, false, false, lr);
            DP dp = new DP(deviceIds, 0, networkType, parameters, pdl, 2);
            dp.load("/omega/models/llama3-26-fullsft.model");
            dp.train();
            String save_model_path = "/omega/models/llama3-sft-1024.model";
            ModelUtils.saveModel((Llama3) dp.getMaster(), save_model_path);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void omega_r1_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int max_len = 1024;
            int embedDim = 512;
            int headNum = 8;
            int nKVHeadNum = 2;
            int decoderNum = 16;
            int vocabSize = 6400;
            String vocabPath = "D:\\test\\models\\mmdit\\6400_tokenizer\\vocab.json";
            String mergesPath = "D:\\test\\models\\mmdit\\6400_tokenizer\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, vocabSize, max_len, embedDim, bias, dropout, flashAttention);
            String model_path = "D:\\test\\models\\mmdit\\llama3-deepseek-distill-1024.model";
            ModelUtils.loadModel(network, model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            Tensor testInput = null;
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                String qaStr = tokenizer.sos_str() + "user\n" + input_txt + tokenizer.eos_str() + "\n";
                //				System.out.println(qaStr);
                int[] idx = tokenizer.encodeInt(qaStr);
                int startLen = idx.length;
                Tensor input = Llama3Test.loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = Llama3Test.output2NextIDXTopN(output, idx.length - 1, 8, network.cudaManager);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos) {
                        break;
                    }
                    input = Llama3Test.loadByTxtToIdx(testInput, idx);
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
    
    public static void testBinData() {
        try {
            int batchSize = 128;
            int max_len = 1024;
            String trainPath = "E:\\dataset\\sft_1024_6400.bin";
            String vocabPath = "E:\\dataset\\6400_tokenizer\\vocab.json";
            String mergesPath = "E:\\dataset\\6400_tokenizer\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTBinDataset trainData = new SFTBinDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            //			Tensor input = new Tensor(batchSize * max_len, 1, 1, 1, true);
            //
            //			float[] tmpInput = new float[batchSize * max_len];
            //
            //			Tensor label = new Tensor(batchSize , 1, 1, max_len, true);
            //
            //			float[] tmpLabel = new float[batchSize * max_len];
            //
            //			int[] padCount = new int[] {0};
            //			for(int i = 0;i<400000;i++) {
            //				trainData.loadData(input, label, tmpInput, tmpLabel, padCount, i);
            //				System.err.println(i);
            //			}
            int[] rankIds = new int[]{0, 1, 2, 3};
            ParallelDataLoader pdl = new ParallelDataLoader(trainData, rankIds);
            ExecutorService executorService = Executors.newFixedThreadPool(rankIds.length);
            for (int rankId : rankIds) {
                ThreadDataset td = pdl.getDataloaders().get(rankId);
                Tensor input = new Tensor(batchSize * max_len, 1, 1, 1, true);
                float[] tmpInput = new float[batchSize * max_len];
                Tensor label = new Tensor(batchSize, 1, 1, max_len, true);
                float[] tmpLabel = new float[batchSize * max_len];
                int[] padCount = new int[]{0};
                executorService.execute(new Runnable() {
                    @Override
                    public void run() {
                        // TODO Auto-generated method stub
                        try {
                            td.loadData(input, label, tmpInput, tmpLabel, padCount, 0);
                            for (int i = 0; i < 10000; i++) {
                                //								long start = System.nanoTime();
                                System.out.println(rankId + ":" + i);
                                td.loadData(input, label, tmpInput, tmpLabel, padCount, i);
                                //								System.out.println((System.nanoTime() - start)/1e6+"ms.");
                                //								input.showDM();
                            }
                            //							System.out.println(JsonUtils.toJson(tmpInput));
                            //
                            //							input.showDM();
                        } catch (Exception e) {
                            // TODO: handle exception
                            e.printStackTrace();
                        }
                    }
                });
            }
            executorService.shutdown();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static int output2NextIDX(Tensor output, int nextTokenIdx) {
        if (nextTokenIdx < output.number) {
            return pickTopN(output.getByNumber(nextTokenIdx), 1);
        }
        return 0;
    }

    //	public static int output2NextIDXTopN(Tensor output,int nextTokenIdx,int topK) {
    //		SoftmaxKernel kernel = new SoftmaxKernel();
    //		Tensor tmp = new Tensor(1, 1, 1, output.width, true);
    //		Tensor prof = new Tensor(1, 1, 1, output.width, true);
    //		if(nextTokenIdx < output.number) {
    //			tmp.hostToDevice(MatrixOperation.multiplication(output.getByNumber(nextTokenIdx), 0.7f));
    //			kernel.softmax_out(tmp, prof);
    //			return pickTopN(prof.syncHost(), topK);
    //		}
    //		return 0;
    //	}
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

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            //			train_pretrain();
            //			train_full_sft();
            //			testBinData();
            //			dp_train_pretrain();
//            dp_train_full_sft();
            //			dp_train_1024_sft();
            omega_r1_predict();           
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
