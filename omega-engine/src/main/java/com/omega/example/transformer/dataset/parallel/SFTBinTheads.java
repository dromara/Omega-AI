package com.omega.example.transformer.dataset.parallel;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.example.transformer.dataset.parallel.params.DataLoaderParamters;
import com.omega.example.transformer.dataset.parallel.params.SFTBinParamters;
import com.omega.example.transformer.utils.bpe.BinDataType;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import jcuda.runtime.JCuda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.RandomAccessFile;
import java.util.concurrent.CompletableFuture;

public class SFTBinTheads extends ThreadDataset {
    public int max_len = 256;
    public Tensor testInput;
    private String dataPath;
    private CompletableFuture<Boolean> cf;
    private FileReader fileReader;
    private BufferedReader bufferedReader;
    private RandomAccessFile file;
    private BinDataType dataType = BinDataType.unint32;
    private long index = 0;
    private int[] cache = null;
    private short[] cacheShort = null;
    private int[] tmpCount = new int[]{0};
    private int byteUnit = 4;
    private int rankId;
    private long partSize;
    private long skip;

    public SFTBinTheads(String dataPath, int max_len, int batchSize, int rankId, long partSize, Tokenizer tokenizer, BinDataType dataType) {
        this.tokenizer = tokenizer;
        this.dataType = dataType;
        this.rankId = rankId;
        this.partSize = partSize;
        if (dataType == BinDataType.unint16) {
            byteUnit = 2;
        }
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.setBatchSize(batchSize);
        this.number = loadBinCount();
        initBinReader();
        this.count_it = this.number / batchSize;
        //		System.out.println("dataCount:"+this.number);
        //		System.out.println("count_it:"+this.count_it);
    }

    public void initBinReader() {
        try {
            index = 0;
            skip = rankId * partSize * max_len * byteUnit;
            System.err.println(skip);
            file.seek(skip);
            System.out.println("dataset[" + rankId + "] is ready.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int loadBinCount() {
        try {
            file = new RandomAccessFile(dataPath, "r");
            number = (int) (file.length() / max_len / byteUnit);
            cache = new int[max_len * getBatchSize()];
            cacheShort = new short[max_len * getBatchSize()];
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
            if (fileReader != null) {
                fileReader.close();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] loadData() {
        try {
            if ((index + 1) * max_len * byteUnit <= partSize * max_len * byteUnit) {
                //				System.out.println(index);
                if (dataType == BinDataType.unint16) {
                    ModelUtils.readShort2Int(file, cache);
                } else {
                    ModelUtils.loadIntData(file, cache);
                }
                file.seek(file.getFilePointer() - byteUnit);
                index++;
            } else {
                initBinReader();
                return loadData();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
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
                //				int number = 0;
                //				for(int b = 0;b<getBatchSize();b++) {
                //					int[] onceToken = loadData();
                //					int count = formatToIdx(b, onceToken, input, label);
                //					number += count;
                //				}
                int[] onceToken = loadBatchData();
                int number = formatToIdx(onceToken, input, label);
                counts[0] = number;
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }

    public int[] loadBatchData() {
        try {
            long current = (index + 1) * getBatchSize() * max_len * byteUnit + skip;
            //			System.err.println(current+"<="+file.length());
            if (current <= file.length()) {
                if (dataType == BinDataType.unint16) {
                    ModelUtils.readShort2Int(file, cache, cacheShort);
                } else {
                    ModelUtils.loadIntData(file, cache);
                }
                index++;
            } else {
                initBinReader();
                return loadData();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return cache;
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

    public int formatToIdx(int[] onceToken, float[] input, float[] label) {
        int number = 0;
        //		System.out.println(JsonUtils.toJson(onceToken));
        for (int b = 0; b < getBatchSize(); b++) {
            for (int t = 0; t < max_len; t++) {
                int curr = onceToken[b * max_len + t];
                int next = tokenizer.eos();
                if (t + 1 < max_len) {
                    next = onceToken[b * max_len + t + 1];
                } else if (t + 1 >= max_len && (curr == tokenizer.pad() || curr == tokenizer.eos())) {
                    next = tokenizer.pad();
                }
                if (next != tokenizer.pad()) {
                    number++;
                }
                input[b * max_len + t] = curr;
                label[b * max_len + t] = next;
            }
        }
        //		System.out.println(JsonUtils.toJson(label));
        return number;
    }

    @Override
    public void loadData(DataLoaderParamters params) {
        // TODO Auto-generated method stub
        SFTBinParamters p = (SFTBinParamters) params;
        this.loadData(p.getInput(), p.getLabel(), p.getTmpInput(), p.getTmpLabel(), p.getPadCount(), p.getIt());
    }

    @Override
    public DataLoaderParamters createParamters(Network network) {
        // TODO Auto-generated method stub
        Tensor input = new Tensor(getBatchSize() * network.time, 1, 1, 1, true);
        float[] tmpInput = new float[getBatchSize() * network.time];
        Tensor label = new Tensor(getBatchSize(), 1, 1, network.time, true);
        float[] tmpLabel = new float[getBatchSize() * network.time];
        int[] padCount = new int[]{0};
        Llama3 net = (Llama3) network;
        Tensor[] cs = RoPEKernel.getCosAndSin(network.time, net.embedDim, net.headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        SFTBinParamters p = new SFTBinParamters(input, label, tmpInput, tmpLabel, padCount, 0, cos, sin);
        return p;
    }
}

