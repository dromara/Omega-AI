package com.omega.example.transformer.test;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.llama.LlamaCausalSelfAttention2Layer;
import com.omega.engine.nn.layer.llama.LlamaMLPLayer;
import com.omega.engine.nn.layer.llama.LlamaTransformerBlock;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.dataset.PreTrainDataset;
import com.omega.example.transformer.tokenizer.bertTokenizer.BertTokenizer;
import com.omega.example.transformer.utils.*;
import com.omega.example.transformer.utils.bpe.BinDataType;

import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

public class Llama2Test {
    public static void test_llama2() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 2;
            int max_len = 4;
            int embedDim = 12;
            int decoderNum = 4;
            int head_num = 2;
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, 100, max_len, embedDim, bias, dropout, false);
            Tensor x = new Tensor(batchSize * max_len, 1, 1, 1, MatrixUtils.order(batchSize * max_len, 1, 1), true);
            Tensor y = new Tensor(batchSize * max_len, 1, 1, 1, MatrixUtils.order(batchSize * max_len, 1, 1), true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            cos.showDM();
            sin.showDM();
            CUDAModules.initCUDAFunctions();
            Tensor output = network.forward(cos, sin, x);
            System.err.println("output:");
            output.showDM();
            Tensor loss = network.loss(output, y, -1);
            System.err.println("loss:");
            loss.showDM();
            float lossV = MatrixOperation.sum(loss.syncHost()) / (batchSize * max_len);
            System.out.println(lossV);
            Tensor lossDiff = network.lossDiff(output, y, -1);
            System.err.println("lossDiff:");
            lossDiff.showDM();
            network.back(cos, sin, lossDiff);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_yl_qa() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 16;
            int max_len = 128;
            int embedDim = 768;
            int head_num = 12;
            int decoderNum = 6;
            String trainPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\qaData.txt";
            CNChatTokenizer trainData = new CNChatTokenizer(trainPath, max_len, batchSize);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
            network.learnRate = 0.0002f;
            //			String model_path = "H:\\model\\llama2-110m-qa.model";
            //
            //			ModelUtils.loadModel(network, model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 3, 0.0001f, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{1, 2};
            optimizer.trainLlama2(trainData);
            String out_model_path = "H:\\model\\llama2-110m-qa.model";
            ModelUtils.saveModel(network, out_model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase() + " ";
                System.out.println("user:" + input_txt);
                int[] idx = trainData.encode(input_txt);
                int startLen = idx.length;
                Tensor input = trainData.loadByTxtToIdx(idx);
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
                    if (nextIDX == trainData.dictionary.get("<eos>").intValue() || nextIDX == trainData.dictionary.get("<sep>").intValue()) {
                        break;
                    }
                    input = trainData.loadByTxtToIdx(idx);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                System.out.println("chatbot:" + trainData.decode(idx, startLen));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_yl_qa_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 16;
            int max_len = 128;
            int embedDim = 768;
            int head_num = 12;
            int decoderNum = 6;
            String trainPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\qaData.txt";
            CNChatTokenizer trainData = new CNChatTokenizer(trainPath, max_len, batchSize);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
            String model_path = "H:\\model\\llama2-110m-qa.model";
            ModelUtils.loadModel(network, model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase() + " ";
                System.out.println("user:" + input_txt);
                int[] idx = trainData.encode(input_txt);
                int startLen = idx.length;
                Tensor input = trainData.loadByTxtToIdx(idx);
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDX(output, idx.length - 1, 1);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == trainData.dictionary.get("<eos>").intValue() || nextIDX == trainData.dictionary.get("<sep>").intValue()) {
                        break;
                    }
                    input = trainData.loadByTxtToIdx(idx);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                System.out.println("chatbot:" + trainData.decode(idx, startLen));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_wiki_chatglm_vocab() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 4;
            int max_len = 256;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\wiki_idx_chatglm_voc.txt";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            CNWikiTokenizer4 trainData = new CNWikiTokenizer4(trainPath, max_len, batchSize, 254547, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
            optimizer.lr_step = new int[]{1, 2};
            optimizer.lr = 3e-4f;
            optimizer.min_lr = 1e-5f;
            optimizer.setWarmUp(true);
            optimizer.warmUpTime = 1000;
            optimizer.lrDecayIters = (int) (trainData.count_it * 0.96);
            optimizer.trainLlama2_chinese(trainData);
            String model_path = "H:\\model\\llama2-92m-chinese.model";
            ModelUtils.saveModel(network, model_path);
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
                Tensor input = trainData.loadByTxtToIdx(idx);
                input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - idx.length; t++) {
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

    public static void llama2_cn_wiki_smallvocab() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 256;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            int maxDataCount = 1000000;
            String trainPath = "H:\\transformer_dataset\\wiki_idx_smallvocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\vocab.txt";
            BertTokenizer tokenizer = new BertTokenizer(tokenizer_path, true, true);
            CNWikiTokenizer2 trainData = new CNWikiTokenizer2(trainPath, max_len, batchSize, tokenizer, maxDataCount);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 0.0001f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {1, 2};
            //			optimizer.lr = 3e-4f;
            //			optimizer.min_lr = 1e-5f;
            //			optimizer.setWarmUp(true);
            //			optimizer.warmUpTime = 1000;
            //			optimizer.lrDecayIters = 30000;
            optimizer.trainLlama2_wiki(trainData);
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
                int[] idx = tokenizer.encode(input_txt);
                Tensor input = trainData.loadByTxtToIdx(idx);
                input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - idx.length; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
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

    public static void llama2_chinese_smallvocab() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 256;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\wbm_idx_smallvocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\vocab.txt";
            BertTokenizer tokenizer = new BertTokenizer(tokenizer_path, true, true);
            CNWikiTokenizer3 trainData = new CNWikiTokenizer3(trainPath, max_len, batchSize, 6250865, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
            optimizer.lr_step = new int[]{1, 2};
            optimizer.lr = 3e-4f;
            optimizer.min_lr = 1e-5f;
            optimizer.setWarmUp(true);
            optimizer.warmUpTime = 1000;
            optimizer.lrDecayIters = (int) (trainData.count_it * 0.96);
            optimizer.trainLlama2_chinese(trainData);
            String model_path = "H:\\model\\llama2-92m-chinese.model";
            ModelUtils.saveModel(network, model_path);
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
                int[] idx = tokenizer.encode(input_txt);
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

    public static void llama2_chinese_smallvocab_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 256;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\wbm_idx_smallvocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\vocab.txt";
            BertTokenizer tokenizer = new BertTokenizer(tokenizer_path, true, true);
            CNWikiTokenizer3 trainData = new CNWikiTokenizer3(trainPath, max_len, batchSize, 6250865, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String model_path = "H:\\transformer_dataset\\llama2-92m-chinese.model";
            ModelUtils.loadModel(network, model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            int max_test = 100;
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                System.out.println("user:" + input_txt);
                int[] idx = tokenizer.encode(input_txt);
                int startLen = idx.length;
                Tensor input = trainData.loadByTxtToIdx(idx, max_test);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(max_test, network.embedDim, network.headNum);
                for (int t = 0; t < max_test - startLen; t++) {
                    network.time = max_test;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDX(output, idx.length - 1, 5);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos || nextIDX == tokenizer.pad) {
                        break;
                    }
                    input = trainData.loadByTxtToIdx(idx, max_test);
                    //					RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                System.out.println("chatbot:" + tokenizer.decode(idx));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_chinese_chatglm_vocab() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 2;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\wbm_idx_chatglm_vocab.bin";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            PreTrainDataset trainData = new PreTrainDataset(trainPath, max_len, batchSize, tokenizer, BinDataType.unint32);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 2e-4f;
            initWeight(network, decoderNum);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.CONSTANT, false);
            optimizer.trainLlama2_chinese(trainData, 8, true);
            String model_path = "H:\\model\\llama2-92m-chinese2.model";
            ModelUtils.saveModel(network, model_path);
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

    public static void llama2_chat_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\sft_data_chatglm_idx_vocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            CNWikiTokenizer4 trainData = new CNWikiTokenizer4(trainPath, max_len, batchSize, 6250865, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String pertrain_model_path = "H:\\model\\llama2-110m-chinese.model";
            ModelUtils.loadModel(network, pertrain_model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
            optimizer.lr_step = new int[]{1, 2};
            optimizer.lr = 3e-4f;
            optimizer.min_lr = 1e-5f;
            optimizer.setWarmUp(true);
            optimizer.warmUpTime = 1000;
            optimizer.lrDecayIters = (int) (trainData.count_it * 0.96);
            optimizer.trainLlama2_chinese_sft(trainData);
            String model_path = "H:\\model\\llama2-110m-chinese.model";
            ModelUtils.saveModel(network, model_path);
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

    public static void llama2_medical_sft() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\sft_data_chatglm_idx_vocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            CNWikiTokenizer4 trainData = new CNWikiTokenizer4(trainPath, max_len, batchSize, 6250865, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String pertrain_model_path = "H:\\model\\llama2-110m-chinese.model";
            ModelUtils.loadModel(network, pertrain_model_path);
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
            optimizer.lr_step = new int[]{1, 2};
            optimizer.lr = 3e-4f;
            optimizer.min_lr = 1e-5f;
            optimizer.setWarmUp(true);
            optimizer.warmUpTime = 1000;
            optimizer.lrDecayIters = (int) (trainData.count_it * 0.96);
            optimizer.trainLlama2_chinese_sft(trainData);
            String model_path = "H:\\model\\llama2-110m-chinese.model";
            ModelUtils.saveModel(network, model_path);
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

    public static void llama2_chinese_chatglm_vocab_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            boolean flashAttention = false;
            int batchSize = 8;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\wbm_idx_smallvocab.txt";
            String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64793);
            CNWikiTokenizer4 trainData = new CNWikiTokenizer4(trainPath, max_len, batchSize, 6250865, tokenizer);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            network.learnRate = 3e-4f;
            String model_path = "H:\\transformer_dataset\\llama2-110m-chinese-chat.model";
            ModelUtils.loadModel(network, model_path);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            int max_test = 200;
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                System.out.println("user:" + input_txt);
                int[] idx = tokenizer.encodeInt(input_txt);
                idx = Arrays.copyOf(idx, idx.length + 1);
                idx[idx.length - 1] = tokenizer.bos;
                int startLen = idx.length;
                Tensor input = trainData.loadByTxtToIdx(idx, max_test);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(max_test, network.embedDim, network.headNum);
                for (int t = 0; t < max_test - startLen; t++) {
                    network.time = max_test;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = output2NextIDX(output, idx.length - 1, 1);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos || nextIDX == tokenizer.pad) {
                        break;
                    }
                    input = trainData.loadByTxtToIdx(idx, max_test);
                    //					RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                }
                System.out.println("chatbot:" + tokenizer.decode(idx));
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_cn_baike() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 4;
            int max_len = 512;
            int embedDim = 512;
            int head_num = 8;
            int decoderNum = 8;
            int maxDataCount = 1000000;
            String trainPath = "H:\\transformer_dataset\\563w_baidubaike.json";
            CNWikiTokenizer trainData = new CNWikiTokenizer(trainPath, max_len, batchSize, true, maxDataCount);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
            network.learnRate = 3e-4f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
            optimizer.lr = 3e-4f;
            optimizer.min_lr = 1e-5f;
            optimizer.setWarmUp(true);
            optimizer.warmUpTime = 1000;
            optimizer.lrDecayIters = 30000;
            optimizer.trainLlama2_wiki(trainData);
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入中文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase() + " ";
                System.out.println("user:" + input_txt);
                Tensor input = trainData.loadByTxtToIdx(input_txt);
                //				input.showDM();
                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                Tensor cos = pos[0];
                Tensor sin = pos[1];
                //				positions.showDM();
                //				Tensor mask = CNChatTokenizer.triu(1, network.headNum, input.number, input.number, 1);
                //				mask.showDM();
                for (int t = 0; t < max_len; t++) {
                    network.time = input.number;
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    //					output.showDM();
                    String txts = output2WikiTXT(output, trainData, true);
                    //					System.out.println("output:"+txts);
                    String nextWord = txts.substring(txts.length() - 1, input_txt.length());
                    //					System.out.println("nextWord:"+nextWord);
                    if (trainData.sd.get(nextWord) != null && (trainData.sd.get(nextWord).equals("<sep>") || trainData.sd.get(nextWord).equals("<eos>"))) {
                        input_txt += trainData.sd.get(nextWord);
                        break;
                    } else {
                        input_txt += nextWord;
                    }
                    input = trainData.loadByTxtToIdx(input_txt);
                    RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
                    //					CNChatTokenizer.triu(1, network.headNum, input.number, input.number, 1, mask);
                }
                System.out.println("chatbot:" + input_txt.split(" ")[1]);
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void llama2_dp() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 32;
            int max_len = 128;
            int embedDim = 128;
            int headNum = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\gpt\\dpcc50.txt";
            CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
            //			network.CUDNN = true;
            network.learnRate = 0.001f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.001f, LearnRateUpdate.GD_GECAY, false);
            //			optimizer.lr_step = new int[] {20,50,80};
            optimizer.trainLlama2_GEN(trainData);
            String out_model_path = "H:\\model\\llama2-dp.model";
            ModelUtils.saveModel(network, out_model_path);
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

    public static void llama2_dp_predict() {
        try {
            boolean bias = false;
            boolean dropout = false;
            int batchSize = 32;
            int max_len = 128;
            int embedDim = 128;
            int headNum = 8;
            int decoderNum = 8;
            String trainPath = "H:\\transformer_dataset\\gpt\\dpcc50.txt";
            CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
            String model_path = "H:\\model\\llama2-dp.model";
            ModelUtils.loadModel(network, model_path);
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

    public static String output2TXT(Tensor output, CNChatTokenizer trainData, boolean format) {
        String txt = "";
        for (int i = 0; i < output.number; i++) {
            int charIndex = pickTopN(output.getByNumber(i), 1);
            String c = trainData.vocab[charIndex];
            txt += c;
        }
        //		System.out.println("output txt:"+txt);
        if (format) {
            for (String key : trainData.specials_dictionary.keySet()) {
                txt = txt.replaceAll(key, trainData.specials_dictionary.get(key));
            }
        }
        return txt;
    }

    public static String output2WikiTXT(Tensor output, CNWikiTokenizer trainData, boolean format) {
        String txt = "";
        if (trainData.tokenizer != null) {
            int[] idx = new int[output.number];
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                idx[i] = charIndex;
            }
            txt = trainData.tokenizer.decode(idx);
        } else {
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                String c = trainData.vocab[charIndex];
                txt += c;
            }
            //			System.out.println("output txt:"+txt);
            if (format) {
                for (String key : trainData.specials_dictionary.keySet()) {
                    txt = txt.replaceAll(key, trainData.specials_dictionary.get(key));
                }
            }
        }
        return txt;
    }

    public static String output2WikiTXT(Tensor output, CNWikiTokenizer2 trainData, boolean format) {
        String txt = "";
        if (trainData.tokenizer != null) {
            int[] idx = new int[output.number];
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                idx[i] = charIndex;
            }
            txt = trainData.tokenizer.decode(idx);
        } else {
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                String c = trainData.vocab[charIndex];
                txt += c;
            }
            //			System.out.println("output txt:"+txt);
            if (format) {
                for (String key : trainData.specials_dictionary.keySet()) {
                    txt = txt.replaceAll(key, trainData.specials_dictionary.get(key));
                }
            }
        }
        return txt;
    }

    public static int[] output2WikiIDX(Tensor output, CNWikiTokenizer2 trainData) {
        int[] idx = new int[output.number];
        if (trainData.tokenizer != null) {
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                idx[i] = charIndex;
            }
        } else {
            for (int i = 0; i < output.number; i++) {
                int charIndex = pickTopN(output.getByNumber(i), 1);
                idx[i] = charIndex;
            }
        }
        return idx;
    }

    public static int output2NextIDX(Tensor output, int nextTokenIdx) {
        if (nextTokenIdx < output.number) {
            return pickTopN(output.getByNumber(nextTokenIdx), 3);
        }
        return 0;
    }

    public static int output2NextIDX(Tensor output, int nextTokenIdx, int topK) {
        if (nextTokenIdx < output.number) {
            return pickTopN(output.getByNumber(nextTokenIdx), topK);
        }
        return 0;
    }

    public static String genTxt(Tensor input, Tensor output, Llama2 network, CNTokenizer trainData, int time, Tensor[] pos) {
        network.time = input.number;
        RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
        output = network.forward(pos[0], pos[1], input);
        output.syncHost();
        //		output.showDMByNumber(0);
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

    public static void initParams(Tensor p, float mean, float std) {
        p.setData(RandomUtils.normal_(p.dataLength, mean, std));
    }

    public static void initWeight(Llama2 network, int n_layers) {
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

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            //			llama2_yl_qa();
            //			llama2_yl_qa_predict();
            //			llama2_dp();
            //			llama2_dp_predict();
            //			llama2_wiki_chatglm_vocab();
            //			llama2_cn_wiki_smallvocab();
            //			llama2_cn_baike();
            //			llama2_chinese_smallvocab();
            //			test_llama2();
            llama2_chinese_chatglm_vocab();
            //			llama2_chinese_smallvocab_predict();
            //			llama2_chinese_chatglm_vocab_predict();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

