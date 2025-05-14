//package com.omega.boot.starter.service;
//
//import com.omega.common.data.Tensor;
//import com.omega.common.utils.RandomUtils;
//import com.omega.engine.loss.LossType;
//import com.omega.engine.nn.layer.gpu.RoPEKernel;
//import com.omega.engine.nn.network.Llama2;
//import com.omega.engine.nn.network.RunModel;
//import com.omega.engine.updater.UpdaterType;
//import com.omega.example.transformer.utils.CNWikiTokenizer4;
//import com.omega.example.transformer.utils.ModelUtils;
//import com.omega.example.transformer.utils.SentencePieceTokenizer;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import org.springframework.context.annotation.Bean;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.stereotype.Component;
//
//import java.util.Arrays;
//
//
//@Component
//@Configuration
//public class LLama2Service extends ModelAbstract{
//
//    private Logger logger = LoggerFactory.getLogger(LLama2Service.class);
//
//    boolean bias = false;
//    boolean dropout = false;
//    boolean flashAttention = false;
//    int batchSize = 8;
//    int max_len = 512;
//    int embedDim = 512;
//    int head_num = 8;
//    int decoderNum = 8;
//    String trainPath = "H:\\transformer_dataset\\wbm_idx_smallvocab.txt";
//    String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
//
//    String model_path = "H:\\transformer_dataset\\llama2-110m-chinese-chat.model";
//
//    int max_test = 200;
//
//    @Bean
//    public SentencePieceTokenizer getTokenizer() {
//        try {
//            return new SentencePieceTokenizer(tokenizer_path, 64793);
//        } catch (Exception e) {
//            logger.error("Error loading tokenizer: {}", e);
//        }
//        return null;
//    }
//
//    @Bean
//    public CNWikiTokenizer4 getTrainData() {
//        try {
//            return new CNWikiTokenizer4(trainPath, max_len, batchSize, 6250865, getTokenizer());
//        } catch (Exception e) {
//            logger.error("Error loading tokenizer: {}", e);
//        }
//        return null;
//    }
//    @Bean
//    public Llama2 getNetwork() {
//        try {
//
//            Llama2 network = new Llama2(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, decoderNum, getTrainData().vocab_size, max_len, embedDim, bias, dropout, flashAttention);
//            network.learnRate = 3e-4f;
//            ModelUtils.loadModel(network, model_path);
//            network.RUN_MODEL = RunModel.TEST;
//            return network;
//        } catch (Exception e) {
//            logger.error("Error loading llama2: {}", e);
//        }
//        return null;
//    }
//
//    public String predict(String input_txt){
//        try {
//            Llama2 network = getNetwork();
//            SentencePieceTokenizer tokenizer = getTokenizer();
//            CNWikiTokenizer4 trainData = getTrainData();
//            logger.info("请输入中文:");
//            input_txt = input_txt.toLowerCase();
//            logger.info("user:" + input_txt);
//            int[] idx = getTokenizer().encodeInt(input_txt);
//            idx = Arrays.copyOf(idx, idx.length + 1);
//            idx[idx.length - 1] = getTokenizer().bos;
//            int startLen = idx.length;
//            Tensor input = getTrainData().loadByTxtToIdx(idx, max_test);
//            //				input.showDM();
//            Tensor[] pos = RoPEKernel.getCosAndSin(max_test, network.embedDim, network.headNum);
//            for (int t = 0; t < max_test - startLen; t++) {
//                network.time = max_test;
//                Tensor cos = pos[0];
//                Tensor sin = pos[1];
//                Tensor output = network.forward(cos, sin, input);
//                output.syncHost();
//                int nextIDX = output2NextIDX(output, idx.length - 1, 1);
//                idx = Arrays.copyOf(idx, idx.length + 1);
//                idx[idx.length - 1] = nextIDX;
//                if (nextIDX == tokenizer.eos || nextIDX == tokenizer.pad) {
//                    break;
//                }
//                input = trainData.loadByTxtToIdx(idx, max_test);
//                //					RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
//            }
//            String result = tokenizer.decode(idx);
//            logger.info("chatbot:" + result);
//            return result;
//        } catch (Exception e) {
//            // TODO: handle exception
//            logger.error("llama2 predict failed: {}", e);
//        }
//        return null;
//    }
//
//    public static int output2NextIDX(Tensor output, int nextTokenIdx, int topK) {
//        if (nextTokenIdx < output.number) {
//            return pickTopN(output.getByNumber(nextTokenIdx), topK);
//        }
//        return 0;
//    }
//
//    public static int pickTopN(float[] x, int n) {
//        float[] sort = Arrays.copyOf(x, x.length);
//        Arrays.sort(sort);
//        float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
//        float v = topN[RandomUtils.getRandomNumber(topN)];
//        for (int i = 0; i < x.length; i++) {
//            if (v == x[i]) {
//                return i;
//            }
//        }
//        return 0;
//    }
//}
