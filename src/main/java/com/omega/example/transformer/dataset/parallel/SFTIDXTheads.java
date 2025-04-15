package com.omega.example.transformer.dataset.parallel;

import com.omega.common.config.Tensor;
import com.omega.example.transformer.dataset.JSONDatasetLoader;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import jcuda.runtime.JCuda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.CompletableFuture;

public class SFTIDXTheads extends JSONDatasetLoader {
    public int max_len = 256;
    public int vocab_size;
    public Tensor testInput;
    private int batchSize = 1;
    private String dataPath;
    private CompletableFuture<Boolean> cf;
    private FileInputStream fis;
    private BufferedReader bufferedReader;
    private int[] tmpCount = new int[]{0};
    private int[] cache = null;
    private int rankId;
    private int pageSize;

    public SFTIDXTheads(String dataPath, int max_len, int batchSize, Tokenizer tokenizer, int rankId, int pageSize) {
        this.dataPath = dataPath;
        this.rankId = rankId;
        this.pageSize = pageSize;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.voc_size();
        this.number = loadCount();
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + this.vocab_size);
        System.out.println("count_it:" + this.count_it);
    }

    public void initReader() {
        try {
            fis = new FileInputStream(this.dataPath);
            bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            System.out.println("dataset is ready.");
            cache = new int[max_len + 1];
            bufferedReader.skip(rankId * pageSize);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int loadCount() {
        try {
            initReader();
            while (bufferedReader.readLine() != null) {
                number++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void close() {
        try {
            if (bufferedReader != null) {
                bufferedReader.close();
            }
            if (fis != null) {
                fis.close();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] loadData() throws IOException {
        String line = bufferedReader.readLine();
        if (line == null) {
            close();
            initReader();
            return loadData();
        }
        String[] datas = line.split(" ");
        for (int i = 0; i < datas.length; i++) {
            cache[i] = Integer.parseInt(datas[i]);
        }
        return cache;
    }

    public void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int[] padCount, int it) {
        try {
            //			System.out.println(it);
            if (cf != null) {
                boolean success = cf.get();
                if (success) {
                    input.hostToDevice(tmpInput);
                    label.hostToDevice(tmpLabel);
                    System.arraycopy(tmpLabel, 0, label.data, 0, tmpLabel.length);
                    padCount[0] = tmpCount[0];
                    JCuda.cudaDeviceSynchronize();
                }
                cf = loadAsyncData(tmpInput, tmpLabel, tmpCount);
            } else {
                cf = loadAsyncData(tmpInput, tmpLabel, tmpCount);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] input, float[] label, int[] counts) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                int number = 0;
                for (int b = 0; b < batchSize; b++) {
                    int[] onceToken = loadData();
                    int count = formatToIdx(b, onceToken, input, label);
                    number += count;
                }
                counts[0] = number;
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }

    public int formatToIdx(int b, int[] onceToken, float[] input, float[] label) {
        int number = 0;
        for (int t = 0; t < max_len; t++) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            if (next != tokenizer.pad()) {
                number++;
            }
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        }
        return number;
    }
}

