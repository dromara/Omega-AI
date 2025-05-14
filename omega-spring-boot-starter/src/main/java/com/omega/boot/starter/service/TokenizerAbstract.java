package com.omega.boot.starter.service;

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
    private String vocabPath = "vocab.json";
    private String mergesPath = "merges.txt";
    public Object getTokenizer(String tokenizerClass,String path){
        switch (tokenizerClass.toLowerCase()) {
            case "bpetokenizer":
                return getBPETokenizer3(path);
            default:
                logger.error("tokenizer_class is not support: {}", tokenizerClass);
                break;
        }
        return null;
    }
    public BPETokenizer3 getBPETokenizer3(String path) {
        try {
            return new BPETokenizer3(path+ File.separator+vocabPath, path + File.separator+mergesPath);
        } catch (Exception e) {
            logger.error("Error loading tokenizer: {}", e);
        }
        return null;
    }
}
