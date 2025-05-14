package com.omega.boot.starter.service;


import cn.hutool.json.JSONObject;
import com.omega.boot.starter.entity.ModelData;
import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Map;

/**
 * llama3模型初始化类
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
@Configuration
@ConditionalOnExpression("@modelConfig['llama3'] != null")
public class Llama3Service extends ModelAbstract{

    private Logger logger = LoggerFactory.getLogger(Llama3Service.class);

    private static final String model_type = "llama3";

    private boolean bias = false;
    private boolean dropout = false;
    private boolean flashAttention = false;
    private int max_len = 512;
    private int embedDim = 512;
    private int head_num = 16;
    private int nKVHeadNum = 8;
    private int decoderNum = 8;
    private int vocab_size = 6400;
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

    @Bean("llama3")
    public Llama3 getNetwork() {
        try {
            String path = this.modelData.getPath();
            String name = this.modelData.getConfig().getStr("name");
            Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, vocab_size, max_len, embedDim, bias, dropout, flashAttention);
            ModelUtils.loadModel(network, path+ File.separator + name);
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

            BPETokenizer3 tokenizer = (BPETokenizer3)this.tokenizer;
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
            String result = tokenizer.decode(awIdx).replaceAll("<s>assistant\n", "");
            logger.info(result);
            return result;
        } catch (Exception e) {
            // TODO: handle exception
            logger.error("llama3 predict failed: {}", e);
        }
        return null;
    }
}
