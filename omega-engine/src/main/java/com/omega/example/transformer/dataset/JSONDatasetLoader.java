package com.omega.example.transformer.dataset;

import com.omega.common.data.Tensor;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

public abstract class JSONDatasetLoader {
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public String[] vocab;

    public abstract void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int[] padCount, int it);
}

