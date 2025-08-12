package com.omega.example.dit.dataset;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.CompletableFuture;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.utils.BaseTokenizer;
import com.omega.example.transformer.utils.bpe.BinDataType;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

import jcuda.runtime.JCuda;

public class LatendDataset extends BaseTokenizer {
	
	private String clipDataPath;
	
	public int channel;
	public int height;
	public int width;
	public int clipEmbd;
	
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    private int batchSize = 1;
    private String dataPath;
    private RandomAccessFile file;
    private RandomAccessFile clipFile;
    private int index = 0;
    private boolean isBin = false;
    private float[] cache = null;
    private float[] clip_cache = null;
    private CompletableFuture<Boolean> cf;
    private BinDataType dataType = BinDataType.float32;
    private int byteUnit = 4;

    public LatendDataset(String dataPath, String clipDataPath, int batchSize, int channel, int height, int width, int clipEmbd, BinDataType dataType) {
        this.dataType = dataType;
        if (dataType == BinDataType.unint16) {
            byteUnit = 2;
        }
        this.dataPath = dataPath;
        this.clipDataPath = clipDataPath;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.clipEmbd = clipEmbd;
        this.max_len = channel * height * width;
        this.batchSize = batchSize;
        loadBinCount();
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("count_it:" + this.count_it);
    }

    public static void main(String[] args) {

    }

    public int loadBinCount() {
        try {
            file = new RandomAccessFile(dataPath, "r");
            clipFile = new RandomAccessFile(clipDataPath, "r");
            number = (int) (file.length() / max_len / byteUnit);
            cache = new float[max_len];
            clip_cache = new float[clipEmbd];
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void initBinReader() {
        try {
        	index = 0;
            file.seek(0);
            clipFile.seek(0);
            System.out.println("dataset is ready.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public float[] loadData() {
        try {
            if ((index + 1) * max_len * byteUnit <= file.length()) {
                //				System.out.println(index);
                if (dataType == BinDataType.float32) {
                    ModelUtils.readFloat(file, cache);
                    ModelUtils.readFloat(clipFile, clip_cache);
                }
                file.seek(file.getFilePointer());
                clipFile.seek(clipFile.getFilePointer());
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
    
    public float[] loadData(int idx) {
        try {
            if (idx * max_len * byteUnit <= file.length()) {
            	file.seek(idx * max_len * byteUnit);
                clipFile.seek(idx * clipEmbd * byteUnit);
                if (dataType == BinDataType.float32) {
                    ModelUtils.readFloat(file, cache);
                    ModelUtils.readFloat(clipFile, clip_cache);
                }
                
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

    public void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int it) {
        try {
            //			System.out.println(it);
            if (isBin && it == count_it - 2) {
                initBinReader();
            }
            if (cf != null) {
                boolean success = cf.get();
                input.hostToDevice();
                label.hostToDevice();
                input.syncHost(tmpInput);
                label.syncHost(tmpLabel);
                JCuda.cudaDeviceSynchronize();
                cf = loadAsyncData(input, label);
            } else {
                cf = loadAsyncData(input, label);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void loadData(int[] index,Tensor input, Tensor label, int it) {
        try {
            //			System.out.println(it);
            if (isBin && it == count_it - 2) {
                initBinReader();
            }
            if (cf != null) {
                boolean success = cf.get();
                input.hostToDevice();
                label.hostToDevice();
                JCuda.cudaDeviceSynchronize();
                cf = loadAsyncData(index, input, label);
            } else {
                cf = loadAsyncData(index, input, label);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public float[] readIdxData() throws IOException {
    	 return loadData();
    }
    
    public float[] readIdxData(int idx) throws IOException {
   	 return loadData(idx);
   }
    
    public CompletableFuture<Boolean> loadAsyncData(Tensor input, Tensor label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                    float[] onceToken = readIdxData();
                    float[] clipToken = clip_cache;
                    for (int t = 0; t < max_len; t++) {
                        formatNotHeadToIdx(b, t, max_len, onceToken, input);
                    }
                    for(int t = 0;t < clipEmbd;t++) {
                    	formatNotHeadToIdx(b, t, clipEmbd, clipToken, label);
                    }
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }
    
    public CompletableFuture<Boolean> loadAsyncData(int[] index,Tensor input, Tensor label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                	int idx = index[b];
                    float[] onceToken = readIdxData(idx);
                    float[] clipToken = clip_cache;
                    for (int t = 0; t < max_len; t++) {
                        formatNotHeadToIdx(b, t, max_len, onceToken, input);
                    }
                    for(int t = 0;t < clipEmbd;t++) {
                    	formatNotHeadToIdx(b, t, clipEmbd, clipToken, label);
                    }
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] input, float[] label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                	float[] onceToken = readIdxData();
                	 float[] clipToken = clip_cache;
                	for (int t = 0; t < max_len; t++) {
                		formatNotHeadToIdx(b, t, onceToken, input);
                    }
                	for(int t = 0;t < clipEmbd;t++) {
                    	formatNotHeadToIdx(b, t, clipEmbd, clipToken, label);
                    }
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }
    
    public void formatNotHeadToIdx(int b, int t, float[] onceToken, Tensor input) {
        if (t < onceToken.length) {
            input.data[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, int max_len, float[] onceToken, Tensor input) {
        if (t < onceToken.length) {
            input.data[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, float[] onceToken, float[] input) {
        if (t < onceToken.length) {
            input[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, int max_len, float[] onceToken, float[] input) {
        if (t < onceToken.length) {
            input[b * max_len + t] = onceToken[t];
        }
    }
    
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }
    
}
