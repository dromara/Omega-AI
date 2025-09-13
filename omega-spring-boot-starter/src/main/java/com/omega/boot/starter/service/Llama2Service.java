package com.omega.boot.starter.service;

import cn.hutool.json.JSONObject;
import com.omega.boot.starter.entity.ModelData;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.util.Arrays;
import java.util.Map;


@Configuration
@ConditionalOnExpression("@modelConfig['llama2'] != null")
public class Llama2Service extends ModelAbstract{

    private Logger logger = LoggerFactory.getLogger(Llama2Service.class);

    private static final String model_type = "llama2";

    boolean bias = false;
    boolean dropout = false;
    boolean flashAttention = false;
    int batchSize = 8;
    int max_len = 512;
    int embedDim = 512;
    int head_num = 8;
    int decoderNum = 8;

    int max_test = 200;

    @Value("${model.cudnn:false}")
    private boolean cudnn;

    @Autowired
    @Qualifier("modelConfig")
    private Map<String, ModelData> modelConfig;

    private ModelData modelData;

    private Object tokenizer;

    @PostConstruct
    public void init() {
        this.modelData = modelConfig.get(model_type);
        JSONObject tokenizerConfig =  this.modelData.getTokenizerConfig();
        String tokenizerClass = tokenizerConfig.getStr("tokenizer_class");
        this.tokenizer = getTokenizer(tokenizerClass, modelData.getPath());
    }



    @Bean
    public Llama2 getNetwork() {
        try {
            String path = this.modelData.getPath();
            String name = this.modelData.getConfig().getStr("name");
            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, ((SentencePieceTokenizer)tokenizer).voc_size, max_len, embedDim, bias, dropout, flashAttention);
            ModelUtils.loadModel(network, path+ File.separator + name);
            network.RUN_MODEL = RunModel.TEST;
            network.CUDNN = cudnn;
            return network;
        } catch (Exception e) {
            logger.error("Error loading llama2: {}", e);
        }
        return null;
    }

    public String predict(String input_txt){
        try {
            Llama2 network = getNetwork();
            SentencePieceTokenizer sentencePieceTokenizer = (SentencePieceTokenizer)this.tokenizer;
            logger.info("请输入中文:");
            input_txt = input_txt.toLowerCase();
            logger.info("user:" + input_txt);
            int[] idx = sentencePieceTokenizer.encodeInt(input_txt);
            idx = Arrays.copyOf(idx, idx.length + 1);
            idx[idx.length - 1] = sentencePieceTokenizer.bos;
            int startLen = idx.length;
            Tensor inputTensor = null;
            Tensor input = loadByTxtToIdx(inputTensor, idx);
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
                if (nextIDX == sentencePieceTokenizer.eos || nextIDX == sentencePieceTokenizer.pad) {
                    break;
                }
                //input = loadByTxtToIdx(inputTensor, idx);
                //					RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
            }
            String result = sentencePieceTokenizer.decode(idx);
            logger.info("chatbot:" + result);
            return result;
        } catch (Exception e) {
            // TODO: handle exception
            logger.error("llama2 predict failed: {}", e);
        }
        return null;
    }

}
