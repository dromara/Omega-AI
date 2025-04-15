package com.omega.test.transformer.test;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixOperation;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.llama.LlamaCausalSelfAttention2Layer;
import com.omega.engine.nn.layer.llama.LlamaMLPLayer;
import com.omega.engine.nn.layer.llama.LlamaTransformerBlock;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.data.transformer.PreTrainDataset;
import com.omega.data.transformer.SFTDataset;
import com.omega.models.transformer.CNTokenizer;
import com.omega.models.transformer.ModelUtils;
import com.omega.models.transformer.SentencePieceTokenizer;
import com.omega.models.transformer.bpe.BPETokenizer3;
import com.omega.models.transformer.bpe.BinDataType;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Llama3Test {
    public static void llama3_dp() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 32;
            int max_len = 128;
            int embedDim = 128;
            int headNum = 8;
            int nKVHeadNum = 4;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\gpt\\dpcc50.txt";
            CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
            //			network.CUDNN = true;
            network.learnRate = 0.001f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.001f, LearnRateUpdate.GD_GECAY, false);
            //			optimizer.lr_step = new int[] {20,50,80};
            optimizer.trainLlama3_GEN(trainData);
            int gen_len = 1000;
            network.RUN_MODEL = RunModel.TEST;
            Tensor input = null;
            Tensor output = null;
            String pre_txt = "萧炎";
            input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
            Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
            for (int i = 0; i < gen_len; i++) {
                network.time = input.number;
                //				System.out.println(input.number);
                //				input.showDM();
                String txt = genTxt(input, output, network, trainData, pre_txt.length(), pos);
                //				System.out.println("output txt="+txt);
                if (network.time > 1) {
                    pre_txt += txt.substring(input.number - 1, input.number);
                } else {
                    pre_txt += txt;
                }
                //				System.out.println(pre_txt);
                input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
            }
            System.out.println(pre_txt);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama3_monkey() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 2;
            int max_len = 128;
            int embedDim = 512;
            int head_num = 8;
            int nKVHeadNum = 2;
            int decoderNum = 4;
            String trainPath = "H:\\transformer_dataset\\monkey_idx_6400_all_vocab.bin";
            //			String trainPath = "H:\\model\\pretrain_data_6400.bin";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            PreTrainDataset trainData = new PreTrainDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            //			CNBpeTokenizer trainData = new CNBpeTokenizer(trainPath, max_len, batchSize, tokenizer, BinDataType.unint16);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 1e-4f;
            network.CLIP_GRAD_NORM = true;
            //			String model_path = "H:\\model\\llama3-26m-chinese.model";
            //			ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.trainLlama3_chinese(trainData, 8, true);
            String save_model_path = "H:\\model\\llama3-26m-chinese.model";
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

    public static void llama3_monkey_chatglm() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 2;
            int max_len = 256;
            int embedDim = 512;
            int head_num = 8;
            int nKVHeadNum = 4;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\monkey_idx_64793_vocab.bin";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            PreTrainDataset trainData = new PreTrainDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint32);
            //			CNBpeTokenizer trainData = new CNBpeTokenizer(trainPath, max_len, batchSize, tokenizer, BinDataType.unint32);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 1e-4f;
            initWeight(network, decoderNum);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.trainLlama3_chinese(trainData, 8, false);
            String save_model_path = "H:\\model\\llama3-110m-chinese.model";
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

    public static void llama3_monkey_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 3;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\6400\\sft_data_single.csv";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTDataset trainData = new SFTDataset(trainPath, max_len, batchSize, tokenizer);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 1e-4f;
            //			network.CLIP_GRAD_NORM = true;
            String model_path = "H:\\model\\llama3-26m-chinese.model";
            ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {1, 2, 4};
            optimizer.trainLlama3_chinese_sft(trainData, 8, true);
            String save_model_path = "H:\\model\\llama3-26m-chinese.model";
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
                String qaStr = tokenizer.sos_str() + "user\n" + input_txt + tokenizer.eos_str() + "\n";
                System.out.println(qaStr);
                int[] idx = tokenizer.encodeInt(qaStr);
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

    public static void llama3_monkey_med_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 3;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\medical\\med_sft.csv";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTDataset trainData = new SFTDataset(trainPath, max_len, batchSize, tokenizer);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String model_path = "H:\\model\\llama3-26m-chinese.model";
            ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 5, 0.0001f, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {1, 2, 4};
            optimizer.trainLlama3_chinese_sft(trainData, 8, true);
            String save_model_path = "H:\\model\\llama3-26m-chinese.model";
            ModelUtils.saveModel(network, save_model_path);
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
                int[] idx = tokenizer.encodeInt(qaStr);
                int startLen = idx.length;
                Tensor input = loadByTxtToIdx(testInput, idx);
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static void llama3_monkey_med_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            int vocab_size = 6400;
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            String model_path = "H:\\model\\llama3-26m-chinese-sft-med.model";
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
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static void llama3_monkey_sft_test() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            int vocab_size = 6400;
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            String model_path = "H:\\model\\llama3-26m-chinese-sft2.model";
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
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static void llama3_monkey_sql_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 3;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            String trainPath = "H:\\sql_dataset\\full_CSpider\\CSpider\\train_sql.csv";
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            SFTDataset trainData = new SFTDataset(trainPath, max_len, batchSize, tokenizer);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String model_path = "H:\\model\\llama3-26m-chinese.model";
            ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 5, 0.0001f, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {1, 2, 4};
            optimizer.trainLlama3_chinese_sft(trainData, 8, false);
            String save_model_path = "H:\\model\\llama3-26m-chinese-sql.model";
            ModelUtils.saveModel(network, save_model_path);
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
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static void llama3_monkey_sql_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 16;
            int nKVHeadNum = 8;
            int decoderNum = 8;
            int vocab_size = 6400;
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            String model_path = "H:\\model\\llama3-26m-chinese-sql.model";
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
                Tensor input = loadByTxtToIdx(testInput, idx);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs) {
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public static void loadWeight(Map<String, Object> weightMap, Llama3 network) {
        for (String key : weightMap.keySet()) {
            System.out.println(key);
        }
        loadData(network.getDecoder().getSrc_emb().weight, weightMap.get("tok_embeddings.weight"), "tok_embeddings.weight");
        for (int i = 0; i < 8; i++) {
            LlamaTransformerBlock block = network.getDecoder().getDecoderLayers().get(i);
            LlamaCausalSelfAttention2Layer attn = (LlamaCausalSelfAttention2Layer) block.getAttn();
            loadData(attn.getqLinerLayer().weight, weightMap.get("layers." + i + ".attention.wq.weight"), "layers." + i + ".attention.wq.weight");
            loadData(attn.getkLinerLayer().weight, weightMap.get("layers." + i + ".attention.wk.weight"), "layers." + i + ".attention.wk.weight");
            loadData(attn.getvLinerLayer().weight, weightMap.get("layers." + i + ".attention.wv.weight"), "layers." + i + ".attention.wv.weight");
            loadData(attn.getoLinerLayer().weight, weightMap.get("layers." + i + ".attention.wo.weight"), "layers." + i + ".attention.wo.weight");
            block.getNorm1().gamma = loadData(block.getNorm1().gamma, weightMap.get("layers." + i + ".attention_norm.weight"), 1, "layers." + i + ".attention_norm.weight");
            block.getNorm2().gamma = loadData(block.getNorm2().gamma, weightMap.get("layers." + i + ".ffn_norm.weight"), 1, "layers." + i + ".ffn_norm.weight");
            LlamaMLPLayer mlp = block.getMlp();
            loadData(mlp.getLinear1().weight, weightMap.get("layers." + i + ".feed_forward.w1.weight"), "layers." + i + ".feed_forward.w1.weight");
            loadData(mlp.getLinear2().weight, weightMap.get("layers." + i + ".feed_forward.w2.weight"), "layers." + i + ".feed_forward.w2.weight");
            loadData(mlp.getLinear3().weight, weightMap.get("layers." + i + ".feed_forward.w3.weight"), "layers." + i + ".feed_forward.w3.weight");
        }
        network.getDecoder().getNorm().gamma = loadData(network.getDecoder().getNorm().gamma, weightMap.get("norm.weight"), 1, "norm.weight");
        loadData(network.getFullyLayer().weight, weightMap.get("output.weight"), "output.weight");
    }

    public static void initWeight(Llama3 network, int n_layers) {
        initParams(network.getDecoder().getSrc_emb().weight, 0.0f, 0.02f);
        for (int i = 0; i < n_layers; i++) {
            LlamaTransformerBlock block = network.getDecoder().getDecoderLayers().get(i);
            LlamaCausalSelfAttention2Layer attn = (LlamaCausalSelfAttention2Layer) block.getAttn();
            initParams(attn.getqLinerLayer().weight, 0.0f, 0.02f);
            initParams(attn.getkLinerLayer().weight, 0.0f, 0.02f);
            initParams(attn.getvLinerLayer().weight, 0.0f, 0.02f);
            initParams(attn.getoLinerLayer().weight, 0.0f, (float) (0.02f / Math.sqrt(2 * n_layers)));
            LlamaMLPLayer mlp = block.getMlp();
            initParams(mlp.getLinear1().weight, 0.0f, 0.02f);
            initParams(mlp.getLinear2().weight, 0.0f, 0.02f);
            initParams(mlp.getLinear3().weight, 0.0f, (float) (0.02f / Math.sqrt(2 * n_layers)));
        }
        initParams(network.getFullyLayer().weight, 0.0f, 0.02f);
    }

    public static void initParams(Tensor p, float mean, float std) {
        p.setData(RandomUtils.normal_(p.dataLength, mean, std));
    }

    public static String genTxt(Tensor input, Tensor output, Llama3 network, CNTokenizer trainData, int time, Tensor[] pos) {
        network.time = input.number;
        RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
        output = network.forward(pos[0], pos[1], input);
        output.syncHost();
        return output2TXT(output, trainData);
    }

    public static String output2TXT(Tensor output, CNTokenizer trainData) {
        String txt = "";
        for (int i = 0; i < output.number; i++) {
            int charIndex = pickTopN(output.getByNumber(i), 1);
            char c = trainData.dictionaryData[charIndex];
            txt += c;
        }
        return txt;
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

    public static Tensor createTxtData(Tensor input, String txt, int charDim, Map<Character, Integer> dictionary, int maxLenght) {
        int charLength = txt.length();
        if (txt.length() > maxLenght) {
            charLength = maxLenght;
        }
        char[] charset = new char[charLength];
        int start = txt.length() - maxLenght;
        if (start <= 0) {
            start = 0;
        }
        txt.getChars(start, txt.length(), charset, 0);
        float[] td = new float[charLength];
        for (int i = 0; i < charLength; i++) {
            td[i] = dictionary.get(charset[i]);
        }
        if (input == null || input.number != charset.length) {
            input = Tensor.createTensor(input, charset.length, 1, 1, 1, td, true);
        } else {
            input.data = td;
            input.hostToDevice();
        }
        return input;
    }

    public static int output2NextIDX(Tensor output, int nextTokenIdx) {
        if (nextTokenIdx < output.number) {
            return pickTopN(output.getByNumber(nextTokenIdx), 1);
        }
        return 0;
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

    public static void loadData(Tensor x, Object meta, String key) {
        if (meta != null) {
            int dim = getDim(x);
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                x.showShape();
                System.out.println(dataA.size() + ":" + dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            } else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.data[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }
    }

    public static Tensor loadData(Tensor x, Object meta, int dim, String key) {
        if (meta != null) {
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                x = new Tensor(1, 1, 1, dataA.size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                x = new Tensor(dataA.size(), 1, 1, dataA.get(0).size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
                //				float[][] data = (float[][]) meta;
                //				x.data = MatrixUtils.transform(data);
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            } else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                x = new Tensor(N, C, H, W, true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int c = 0; c < dataA.get(n).size(); c++) {
                        for (int h = 0; h < dataA.get(n).get(c).size(); h++) {
                            for (int w = 0; w < dataA.get(n).get(c).get(h).size(); w++) {
                                x.data[n * x.getOnceSize() + c * x.height * x.width + h * x.width + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
            return x;
        }
        return null;
    }

    public static int getDim(Tensor x) {
        int dim = 0;
        if (x.number > 1) {
            dim++;
        }
        if (x.channel > 1) {
            dim++;
        }
        if (x.height > 1) {
            dim++;
        }
        if (x.width > 1) {
            dim++;
        }
        return dim;
    }

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            //			llama3_dp();
            llama3_monkey();
            //			llama3_monkey_chatglm();
            //			llama3_monkey_sft();
            //			llama3_monkey_sft_test();
            //			llama3_monkey_med_sft();
            //			llama3_monkey_med_predict();
            //			llama3_monkey_sql_sft();
            //			llama3_monkey_sql_predict();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
