package com.omega.data.transformer.bertTokenizer;

import java.util.List;

public interface Tokenizer {
    public List<String> tokenize(String text);
}

