package com.omega.models.transformer.bpe;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixOperation;
import com.omega.utils.PrintUtils;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.models.transformer.BaseTokenizer;
import com.omega.models.transformer.tokenizers.Tokenizer;
import jcuda.runtime.JCuda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.CompletableFuture;

public class CNBpeTokenizer extends BaseTokenizer {
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    private int batchSize = 1;
    private String dataPath;
    private int[] targetLens;
    private int min_len = 30;
    private FileReader fileReader;
    private BufferedReader bufferedReader;
    private RandomAccessFile file;
    private int index = 0;
    private boolean isBin = false;
    private int[] cache = null;
    private CompletableFuture<Boolean> cf;
    private BinDataType dataType = BinDataType.unint32;
    private int byteUnit = 4;

    public CNBpeTokenizer(String dataPath, int max_len, int batchSize, int number, Tokenizer tokenizer, BinDataType dataType) {
        this.dataType = dataType;
        if (dataType == BinDataType.unint16) {
            byteUnit = 2;
        }
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.voc_size();
        initReader();
        this.number = number;
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + this.vocab_size);
        System.out.println("count_it:" + this.count_it);
    }

    public CNBpeTokenizer(String dataPath, int max_len, int batchSize, Tokenizer tokenizer, BinDataType dataType) {
        this.dataType = dataType;
        if (dataType == BinDataType.unint16) {
            byteUnit = 2;
        }
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.voc_size();
        if (dataPath.contains(".bin")) {
            this.isBin = true;
        }
        if (isBin) {
            this.number = loadBinCount();
        } else {
            this.number = loadCount();
        }
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + this.vocab_size);
        System.out.println("count_it:" + this.count_it);
    }

    public static Tensor getPositions(int b, int time) {
        float[] data = new float[b * time];
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                data[n * time + t] = t;
            }
        }
        Tensor positions = new Tensor(b * time, 1, 1, 1, data, true);
        return positions;
    }

    public static Tensor getPositions(int b, int c, int time) {
        float[] data = new float[b * c * time];
        for (int n = 0; n < b * c; n++) {
            int pt = n % c;
            for (int t = 0; t < time; t++) {
                if (pt == t) {
                    data[n * time + t] = 1;
                }
            }
        }
        Tensor positions = new Tensor(b * c, 1, 1, time, data, true);
        return positions;
    }

    public static void getPositions(int b, int c, int time, Tensor positions) {
        positions = Tensor.createTensor(positions, b * time, 1, 1, time, true);
        for (int n = 0; n < b * c; n++) {
            int pt = n % b;
            for (int t = 0; t < time; t++) {
                if (pt == t) {
                    positions.data[n * time + t] = 1;
                }
            }
        }
        positions.hostToDevice();
    }

    public static void getPositions(int b, int time, Tensor positions) {
        positions = Tensor.createTensor(positions, b * time, 1, 1, 1, true);
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                positions.data[n * time + t] = t;
            }
        }
        positions.hostToDevice();
    }

    public static Tensor triu(int b, int h, int size1, int size2, float val) {
        float[] data = new float[b * h * size1 * size2];
        for (int n = 0; n < b; n++) {
            for (int hn = 0; hn < h; hn++) {
                for (int i = 0; i < size1; i++) {
                    for (int j = 0; j < size2; j++) {
                        if (i < j) {
                            data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
                        }
                    }
                }
            }
        }
        Tensor mask = new Tensor(b, h, size1, size2, data, true);
        return mask;
    }

    public static void triu(int b, int h, int size1, int size2, float val, Tensor mask) {
        mask = Tensor.createTensor(mask, b, h, size1, size2, true);
        for (int n = 0; n < b; n++) {
            for (int hn = 0; hn < h; hn++) {
                for (int i = 0; i < size1; i++) {
                    for (int j = 0; j < size2; j++) {
                        if (i < j) {
                            mask.data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public static void triu(float val, int[] targetLens, Tensor mask) {
        for (int n = 0; n < mask.number; n++) {
            for (int hn = 0; hn < mask.channel; hn++) {
                for (int i = 0; i < mask.height; i++) {
                    for (int j = 0; j < mask.width; j++) {
                        //						System.out.println(i+":"+targetLens[n]);
                        if (i < targetLens[n]) {
                            if (i < j) {
                                //								System.out.println(i+":"+j);
                                mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
                            }
                        } else {
                            mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public static void main(String[] args) {
        //		String dataPath = "H:\\transformer_dataset\\gpt\\chatdata\\train1w.txt";
        //
        //		int batchSize = 64;
        //
        //		CNTokenizer tokenizer = new CNTokenizer(dataPath, 256, batchSize);
        //		tokenizer.loadDataForTXT();
        //		int[] targetLens = new int[] {2, 4};
        //		Tensor subsequent_mask = triu(2, 4, 5, 5, 1);
        //		triu(1, targetLens, subsequent_mask);
        //		subsequent_mask.showDM();
        //		PrintUtils.printImage(subsequent_mask);
        Tensor positions = getPositions(2, 3, 4);
        PrintUtils.printImage(positions);
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

    public int loadCount() {
        try {
            fileReader = new FileReader(this.dataPath);
            bufferedReader = new BufferedReader(fileReader);
            while (bufferedReader.readLine() != null) {
                number++;
            }
            bufferedReader.reset();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public int loadBinCount() {
        try {
            file = new RandomAccessFile(dataPath, "r");
            number = (int) (file.length() / max_len / byteUnit);
            cache = new int[max_len + 1];
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void initBinReader() {
        try {
            file.seek(0);
            System.out.println("dataset is ready.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void initReader() {
        try {
            fileReader = new FileReader(this.dataPath);
            bufferedReader = new BufferedReader(fileReader);
            System.out.println("dataset is ready.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] loadData() {
        try {
            if ((index + 1) * max_len * byteUnit <= file.length()) {
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

    public int[] readIdx() throws IOException {
        String line = bufferedReader.readLine();
        if (line == null) {
            close();
            initReader();
            line = bufferedReader.readLine();
        }
        String[] ids = line.split(" ");
        if (ids.length >= min_len) {
            int[] tokens = null;
            if (ids.length > max_len) {
                tokens = new int[max_len];
            } else {
                tokens = new int[ids.length];
            }
            for (int i = 0; i < tokens.length; i++) {
                tokens[i] = Integer.parseInt(ids[i]);
            }
            return tokens;
        } else {
            return readIdx();
        }
    }

    public int[] readBinIdx() throws IOException {
        String line = bufferedReader.readLine();
        if (line == null) {
            close();
            initReader();
            line = bufferedReader.readLine();
        }
        String[] ids = line.split(" ");
        if (ids.length >= min_len) {
            int[] tokens = null;
            if (ids.length > max_len) {
                tokens = new int[max_len];
            } else {
                tokens = new int[ids.length];
            }
            for (int i = 0; i < tokens.length; i++) {
                tokens[i] = Integer.parseInt(ids[i]);
            }
            return tokens;
        } else {
            return readIdx();
        }
    }

    public int[] readIdxData() throws IOException {
        if (isBin) {
            return loadData();
        } else {
            return readIdx();
        }
    }

    public void loadData2(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int it) {
        try {
            //			System.out.println(it);
            if (isBin && it == count_it - 4) {
                initBinReader();
            }
            if (cf != null) {
                boolean success = cf.get();
                //				System.err.println(it+"/"+count_it+":"+success);
                //				System.out.println(JsonUtils.toJson(input.data));
                input.hostToDevice(tmpInput);
                label.hostToDevice(tmpLabel);
                JCuda.cudaDeviceSynchronize();
                //				System.out.println(JsonUtils.toJson(tmpLabel));
                cf = loadAsyncData(tmpInput, tmpLabel);
            } else {
                cf = loadAsyncData(tmpInput, tmpLabel);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void loadData(int[] indexs, Tensor input, Tensor label) {
    //		// TODO Auto-generated method stub
    //
    //		input.clear();
    //		label.clear();
    //
    //		for(int i = 0;i<indexs.length;i++) {
    //			Integer[] onceToken = idxData.get(indexs[i]);
    ////			System.out.println(onceToken.length);
    //			for(int t = 0;t<max_len;t++) {
    //				formatNotHeadToIdx(i, t, onceToken, input, label);
    //			}
    //		}
    //
    //		/**
    //		 * copy data to gpu.
    //		 */
    //		input.hostToDevice();
    //		label.hostToDevice();
    //
    //	}
    public void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int it) {
        try {
            //			System.out.println(it);
            if (isBin && it == count_it - 4) {
                initBinReader();
            }
            if (cf != null) {
                boolean success = cf.get();
                //				System.err.println(it+"/"+count_it+":"+success);
                //				System.out.println(JsonUtils.toJson(input.data));
                input.hostToDevice();
                label.hostToDevice();
                input.syncHost(tmpInput);
                label.syncHost(tmpLabel);
                JCuda.cudaDeviceSynchronize();
                //				System.out.println(JsonUtils.toJson(tmpLabel));
                cf = loadAsyncData(input, label);
            } else {
                cf = loadAsyncData(input, label);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public CompletableFuture<Boolean> loadAsyncData(Tensor input, Tensor label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                    int[] onceToken = readIdxData();
                    if (isBin) {
                        for (int t = 0; t < max_len; t++) {
                            formatNotHeadToIdx(b, t, onceToken, input, label);
                        }
                    } else {
                        for (int t = 0; t < max_len; t++) {
                            formatToIdx(b, t, onceToken, input, label);
                        }
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
                    int[] onceToken = readIdxData();
                    if (isBin) {
                        for (int t = 0; t < max_len; t++) {
                            formatNotHeadToIdx(b, t, onceToken, input, label);
                        }
                    } else {
                        for (int t = 0; t < max_len; t++) {
                            formatToIdx(b, t, onceToken, input, label);
                        }
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

    public String decode(Tensor output) {
        int[] tokens = new int[output.number];
        for (int t = 0; t < output.number; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
            tokens[t] = predictIndex;
        }
        return tokenizer.decode(tokens);
    }

    public Tensor loadByTxtToIdx(String txt) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        for (int t = 0; t < txt.length(); t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs) {
        //		System.out.println(idxs.length);
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    //	public static void getPositions(int b,int time,Tensor positions) {
    //		positions = Tensor.createTensor(positions, b * time, 1, 1, time, true);
    //		for(int n = 0;n<b;n++) {
    //			for(int t = 0;t<time;t++) {
    //				positions.data[n * time * time + t * time + t] = 1;
    //			}
    //		}
    //		positions.hostToDevice();
    //	}
    public Tensor loadByTxtToIdx(String txt, int maxLen) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idx.length; t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs, int maxLen) {
        if (testInput != null) {
            testInput.clear();
            testInput.clearGPU();
        }
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void formatToIdx(int b, int t, int[] onceToken, Tensor input, Tensor label) {
        if (t == 0) {
            input.data[b * max_len + t] = tokenizer.sos();
            label.data[b * max_len + t] = onceToken[t];
        } else if (t < onceToken.length) {
            int curr = onceToken[t - 1];
            int next = onceToken[t];
            input.data[b * max_len + t] = curr;
            label.data[b * max_len + t] = next;
            //			label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
        } else if (t == onceToken.length) {
            int curr = onceToken[t - 1];
            input.data[b * max_len + t] = curr;
            label.data[b * max_len + t] = tokenizer.eos();
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.eos] = 1.0f;
        } else {
            input.data[b * max_len + t] = tokenizer.pad();
            label.data[b * max_len + t] = tokenizer.pad();
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.pad] = 1.0f;
        }
    }

    public void formatToIdx(int b, int t, int[] onceToken, float[] input, float[] label) {
        if (t == 0) {
            input[b * max_len + t] = tokenizer.sos();
            label[b * max_len + t] = onceToken[t];
        } else if (t < onceToken.length) {
            int curr = onceToken[t - 1];
            int next = onceToken[t];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
            //			label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
        } else if (t == onceToken.length) {
            int curr = onceToken[t - 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = tokenizer.eos();
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.eos] = 1.0f;
        } else {
            input[b * max_len + t] = tokenizer.pad();
            label[b * max_len + t] = tokenizer.pad();
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.pad] = 1.0f;
        }
    }

    public void formatNotHeadToIdx(int b, int t, int[] onceToken, Tensor input, Tensor label) {
        if (t < onceToken.length) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input.data[b * max_len + t] = curr;
            label.data[b * max_len + t] = next;
        }
    }

    public void formatNotHeadToIdx(int b, int t, int[] onceToken, float[] input, float[] label) {
        if (t < onceToken.length) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        }
    }

    public int[] getTargetLens() {
        if (targetLens == null || targetLens.length != batchSize) {
            targetLens = new int[batchSize];
        }
        return targetLens;
    }
}

