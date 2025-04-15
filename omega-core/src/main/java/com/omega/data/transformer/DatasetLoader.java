package com.omega.data.transformer;

import com.omega.common.tensor.Tensor;
import com.omega.models.transformer.tokenizers.Tokenizer;

public abstract class DatasetLoader {
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public String[] vocab;

    public abstract void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int[] padCount, int it);
}

