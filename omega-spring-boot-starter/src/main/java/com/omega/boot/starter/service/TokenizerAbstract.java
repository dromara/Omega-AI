package com.omega.boot.starter.service;

import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Tokenizer抽象公共方法类
 *
 * @author haylee
 * @date 2025/05/14 14:33
 */
public class TokenizerAbstract {

    private Logger logger = LoggerFactory.getLogger(TokenizerAbstract.class);
    private String vocab = "vocab.json";
    private String merges= "merges.txt";
    private String tokenizer = "tokenizer.model";
    public Object getTokenizer(String tokenizerClass,String path){
        switch (tokenizerClass.toLowerCase()) {
            case "bpetokenizer3":
                return getBPETokenizer3(path);
            case "sentencepiecetokenizer":
            return getSentencePieceTokenizer(path);
            default:
                logger.error("tokenizer_class is not support: {}", tokenizerClass);
                break;
        }
        return null;
    }
    public BPETokenizer3 getBPETokenizer3(String path) {
        try {
            return new BPETokenizer3(path+ File.separator+vocab, path + File.separator+merges);
        } catch (Exception e) {
            logger.error("Error loading tokenizer: {}", e);
        }
        return null;
    }

    public SentencePieceTokenizer getSentencePieceTokenizer(String path) {
        try {
            return new SentencePieceTokenizer(path+ File.separator+tokenizer, 64793);
        } catch (Exception e) {
            logger.error("Error loading tokenizer: {}", e);
        }
        return null;
    }
}
