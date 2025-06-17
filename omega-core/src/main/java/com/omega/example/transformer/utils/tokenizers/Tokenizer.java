package com.omega.example.transformer.utils.tokenizers;

import java.util.List;

public abstract class Tokenizer {
    public abstract List<Integer> encode(String txt);

    public abstract String decode(List<Integer> ids);

    public abstract String decode(int[] ids);

    public abstract int[] encodeInt(String txt);

    public abstract int[] encodeInt(String txt, int maxLen);

    public abstract int sos();

    public abstract int eos();

    public abstract int pad();

    public abstract String sos_str();

    public abstract String eos_str();

    public abstract String pad_str();

    public abstract int voc_size();
}

