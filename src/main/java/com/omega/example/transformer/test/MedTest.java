package com.omega.example.transformer.test;

import java.io.Console;
import java.util.Arrays;
import java.util.Scanner;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;

public class MedTest {
	
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
            //加载BPE Tokenizer
            String vocabPath = "D:\\models\\llm\\bpe_tokenizer\\vocab.json";
            String mergesPath = "D:\\models\\llm\\bpe_tokenizer\\merges.txt";
            BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
            //定义LLM模型结构
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            //加载预训练模型
            String model_path = "D:\\models\\llm\\llama3-26m-chinese-sft-med.model";
            ModelUtils.loadModel(network, model_path);
            //开启模型推理模式
            network.RUN_MODEL = RunModel.TEST;
            Scanner scanner = new Scanner(System.in);
            Tensor testInput = null;

            while (true) {
                System.out.println("请输入您的问题:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                String qaStr = tokenizer.sos_str() + "user\n" + input_txt + tokenizer.eos_str() + "\n";

                int[] idx = tokenizer.encodeInt(qaStr);
                int startLen = idx.length;
                Tensor input = Llama3Test.loadByTxtToIdx(testInput, idx);

                Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);
                for (int t = 0; t < max_len - startLen; t++) {
                    network.time = input.number;
                    Tensor cos = pos[0];
                    Tensor sin = pos[1];
                    Tensor output = network.forward(cos, sin, input);
                    output.syncHost();
                    int nextIDX = Llama3Test.output2NextIDXTopN(output, idx.length - 1, 5, network.cudaManager);
                    idx = Arrays.copyOf(idx, idx.length + 1);
                    idx[idx.length - 1] = nextIDX;
                    if (nextIDX == tokenizer.eos) {
                        break;
                    }
//                    System.out.print(tokenizer.decode(idx));
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
	
	public static void main(String[] args) {
        try {

            llama3_monkey_med_predict();

        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
	
}
