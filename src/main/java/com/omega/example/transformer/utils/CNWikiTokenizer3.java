package com.omega.example.transformer.utils;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.tokenizer.bertTokenizer.BertTokenizer;
import jcuda.runtime.JCuda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;

public class CNWikiTokenizer3 extends BaseTokenizer {
    public int number = 0;
    public int count_it = 0;
    public BertTokenizer tokenizer;
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
    private boolean loadDataStatus = false;

    public CNWikiTokenizer3(String dataPath, int max_len, int batchSize, int number, BertTokenizer tokenizer) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.vocab_size();
        initReader();
        this.number = number;
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + this.vocab_size);
        System.out.println("count_it:" + this.count_it);
    }

    public CNWikiTokenizer3(String dataPath, int max_len, int batchSize, BertTokenizer tokenizer) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.vocab_size();
        this.number = loadCount();
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

    public void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel) {
        //		System.out.println(loadDataStatus);
        if (this.loadDataStatus) {
            input.hostToDevice();
            label.hostToDevice();
            input.syncHost(tmpInput);
            label.syncHost(tmpLabel);
            JCuda.cudaDeviceSynchronize();
            CompletableFuture.supplyAsync(() -> {
                try {
                    this.loadDataStatus = false;
                    for (int b = 0; b < batchSize; b++) {
                        int[] onceToken = readIdx();
                        for (int t = 0; t < max_len; t++) {
                            formatNotHeadToIdx(b, t, onceToken, input, label);
                        }
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                return "success";
            }).thenAccept(status -> {
                loadDataStatus = true;
            });
        } else {
            CompletableFuture.supplyAsync(() -> {
                try {
                    this.loadDataStatus = false;
                    for (int b = 0; b < batchSize; b++) {
                        int[] onceToken = readIdx();
                        for (int t = 0; t < max_len; t++) {
                            formatNotHeadToIdx(b, t, onceToken, input, label);
                        }
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                return "success";
            }).thenAccept(status -> {
                loadDataStatus = true;
            });
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
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
    public String decode(Tensor output) {
        int[] tokens = new int[output.number];
        for (int t = 0; t < output.number; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
            tokens[t] = predictIndex;
        }
        return tokenizer.decode(tokens);
    }

    public Tensor loadByTxtToIdx(String txt) {
        int[] idx = tokenizer.encode(txt);
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        for (int t = 0; t < txt.length(); t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs) {
        System.out.println(idxs.length);
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(String txt, int maxLen) {
        int[] idx = tokenizer.encode(txt);
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idx.length; t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs, int maxLen) {
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < maxLen; t++) {
            if (t < idxs.length) {
                testInput.data[t] = idxs[t];
            } else {
                testInput.data[t] = 0;
            }
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void formatNotHeadToIdx(int b, int t, int[] onceToken, Tensor input, Tensor label) {
        if (t + 1 < onceToken.length) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input.data[b * max_len + t] = curr;
            label.data[b * max_len + t] = next;
            //			label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
        } else if (t + 1 == onceToken.length) {
            int curr = onceToken[t];
            input.data[b * max_len + t] = curr;
            label.data[b * max_len + t] = tokenizer.eos;
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.eos] = 1.0f;
        } else {
            input.data[b * max_len + t] = tokenizer.pad;
            label.data[b * max_len + t] = tokenizer.pad;
            //			label.data[(b * max_len + t) * vocab_size + tokenizer.pad] = 1.0f;
        }
    }

    public int[] getTargetLens() {
        if (targetLens == null || targetLens.length != batchSize) {
            targetLens = new int[batchSize];
        }
        return targetLens;
    }
}

