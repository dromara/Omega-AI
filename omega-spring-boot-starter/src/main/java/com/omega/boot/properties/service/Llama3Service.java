package com.omega.boot.properties.service;

import com.omega.example.transformer.utils.bpe.BPETokenizer3;

public class Llama3Service {


    public BPETokenizer3 getModelName(String vocabPath, String mergesPath) {
        return new BPETokenizer3(vocabPath, mergesPath);
    }
}
