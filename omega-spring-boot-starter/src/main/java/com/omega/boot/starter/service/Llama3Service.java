package com.omega.boot.starter.service;


import com.omega.boot.starter.properties.ModelProperties;
import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

/**
 * llama3模型初始化类
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */

@Configuration
@EnableConfigurationProperties(ModelProperties.class)
@ConditionalOnExpression("'${model.name}'.equals('llama3') && !'${model.path:}'.isEmpty()")
public class Llama3Service {

    private Logger logger = LoggerFactory.getLogger(Llama3Service.class);
    private boolean bias = false;
    private boolean dropout = false;
    private boolean flashAttention = false;
    private int max_len = 512;
    private int embedDim = 512;
    private int head_num = 16;
    private int nKVHeadNum = 8;
    private int decoderNum = 8;
    private int vocab_size = 6400;
    private String vocabPath = "vocab.json";
    private String mergesPath = "merges.txt";
    private String model_path = "llama3-26m-chinese-sft-med.model";

    @Autowired
    private ModelProperties modelProperties;


    public Llama3Service(){

    }
    @Bean("BPETokenizer3")
    public BPETokenizer3 getTokenizer() {
        try {
            return new BPETokenizer3(modelProperties.getPath()+ File.separator+vocabPath, modelProperties.getPath()+ File.separator+mergesPath);
        } catch (Exception e) {
            logger.error("Error loading tokenizer: {}", e);
        }
        return null;
    }

    @Bean("llama3")
    @ConditionalOnBean(BPETokenizer3.class)
    public Llama3 getNetwork() {
        try {
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            ModelUtils.loadModel(network, modelProperties.getPath()+ File.separator + model_path);
            network.RUN_MODEL = RunModel.TEST;
            return network;
        } catch (Exception e) {
            logger.error("Error loading llama2: {}", e);
        }
        return null;
    }
    public String predict(String input_txt){
        try {
            if(StringUtils.isEmpty(input_txt)){
                return "";
            }

            BPETokenizer3 tokenizer = getTokenizer();
            Llama3 network = getNetwork();

            Tensor testInput = null;

            logger.info("请输入中文:");
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
            String result = "chatbot:" + tokenizer.decode(awIdx).replaceAll("<s>assistant\n", "");
            logger.info(result);
            return result;
        } catch (Exception e) {
            // TODO: handle exception
            logger.error("llama3 predict failed: {}", e);
        }
        return null;
    }

    public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs) {
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
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
}
