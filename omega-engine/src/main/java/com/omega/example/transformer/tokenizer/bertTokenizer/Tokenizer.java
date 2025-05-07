package com.omega.example.transformer.tokenizer.bertTokenizer;

import java.util.List;

public interface Tokenizer {
    public List<String> tokenize(String text);
}

