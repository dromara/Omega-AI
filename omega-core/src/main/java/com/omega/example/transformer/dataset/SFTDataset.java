package com.omega.example.transformer.dataset;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.CsvData;
import cn.hutool.core.text.csv.CsvReader;
import cn.hutool.core.text.csv.CsvRow;
import cn.hutool.core.text.csv.CsvUtil;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class SFTDataset {
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    private int batchSize = 1;
    private String dataPath;
    private int min_len = 30;
    //	private CsvReader reader;
    private List<CsvRow> rows;
    private CompletableFuture<Boolean> cf;
    private int current = 1;

    public SFTDataset(String dataPath, int max_len, int batchSize, Tokenizer tokenizer) {
        this.dataPath = dataPath;
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
                    positions.getData()[n * time + t] = 1;
                }
            }
        }
        positions.hostToDevice();
    }

    public static void getPositions(int b, int time, Tensor positions) {
        positions = Tensor.createTensor(positions, b * time, 1, 1, 1, true);
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                positions.getData()[n * time + t] = t;
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
                            mask.getData()[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public static void triu(float val, int[] targetLens, Tensor mask) {
        for (int n = 0; n < mask.getShape()[0]; n++) {
            for (int hn = 0; hn < mask.getShape()[1]; hn++) {
                for (int i = 0; i < mask.getShape()[2]; i++) {
                    for (int j = 0; j < mask.getShape()[3]; j++) {
                        //						System.out.println(i+":"+targetLens[n]);
                        if (i < targetLens[n]) {
                            if (i < j) {
                                //								System.out.println(i+":"+j);
                                mask.getData()[n * mask.getShape()[1] * mask.getShape()[2] * mask.getShape()[3] + hn * mask.getShape()[2] * mask.getShape()[3] + i * mask.getShape()[2] + j] = val;
                            }
                        } else {
                            mask.getData()[n * mask.getShape()[1] * mask.getShape()[2] * mask.getShape()[3] + hn * mask.getShape()[2] * mask.getShape()[3] + i * mask.getShape()[2] + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public int loadCount() {
        try {
            CsvReader reader = CsvUtil.getReader();
            CsvData data = reader.read(FileUtil.file(this.dataPath));
            rows = data.getRows();
            number = data.getRowCount();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void initReader() {
        current = 1;
    }

    public void loadData(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int it) {
        try {
            //			System.out.println(it);
            if (cf != null) {
                boolean success = cf.get();
                if (success) {
                    //					System.err.println(it+"/"+count_it+":"+success);
                    //					System.out.println(JsonUtils.toJson(input.data));
                    input.hostToDevice(tmpInput);
                    label.hostToDevice(tmpLabel);
                    System.arraycopy(tmpLabel, 0, label.getData(), 0, tmpLabel.length);
                    JCuda.cudaDeviceSynchronize();
                    //					System.out.println(JsonUtils.toJson(tmpLabel));
                }
                cf = loadAsyncData(tmpInput, tmpLabel);
            } else {
                cf = loadAsyncData(tmpInput, tmpLabel);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] readIdx() throws IOException {
        CsvRow csvRow = rows.get(current);
        current++;
        if (current >= number - 1) {
            current = 1;
        }
        String q = csvRow.get(1);
        String a = csvRow.get(2);
        String qaStr = tokenizer.sos_str() + "user\n" + q + tokenizer.eos_str() + "\n" + tokenizer.sos_str() + "assistant\n" + a + tokenizer.eos_str();
        int[] ids = tokenizer.encodeInt(qaStr);
        if (ids.length >= min_len) {
            return ids;
        } else {
            return readIdx();
        }
    }

    public String[] readQA() throws IOException {
        CsvRow csvRow = rows.get(current);
        current++;
        if (current >= number - 1) {
            current = 1;
        }
        String q = csvRow.get(1);
        String a = csvRow.get(2);
        String[] qa = new String[]{q, a};
        return qa;
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] input, float[] label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                    String[] qa = readQA();
                    String q = qa[0];
                    String a = qa[1];
                    String qStr = tokenizer.sos_str() + "user\n" + q + tokenizer.eos_str() + "\n";
                    String qaStr = tokenizer.sos_str() + "user\n" + q + tokenizer.eos_str() + "\n" + tokenizer.sos_str() + "assistant\n" + a + tokenizer.eos_str();
                    int qLen = tokenizer.encodeInt(qStr).length;
                    int[] onceToken = tokenizer.encodeInt(qaStr);
                    for (int t = 0; t < max_len; t++) {
                        formatToIdxPad(b, t, onceToken, qLen, input, label);
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
        int[] tokens = new int[output.getShape()[0]];
        for (int t = 0; t < output.getShape()[0]; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
            tokens[t] = predictIndex;
        }
        return tokenizer.decode(tokens);
    }

    public Tensor loadByTxtToIdx(String txt) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        for (int t = 0; t < txt.length(); t++) {
            testInput.getData()[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs) {
        //		System.out.println(idxs.length);
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.getData()[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(String txt, int maxLen) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idx.length; t++) {
            testInput.getData()[t] = idx[t];
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
            testInput.getData()[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void formatToIdxPad(int b, int t, int[] onceToken, float[] input, float[] label) {
        if (t < onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else if (t == onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else {
            input[b * max_len + t] = tokenizer.pad();
            label[b * max_len + t] = tokenizer.pad();
        }
    }

    public void formatToIdxPad(int b, int t, int[] onceToken, int qLen, float[] input, float[] label) {
        if (t < onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            if (t < qLen - 1) {
                next = tokenizer.pad();
            }
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else if (t == onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            if (t < qLen - 1) {
                next = tokenizer.pad();
            }
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else {
            input[b * max_len + t] = tokenizer.pad();
            label[b * max_len + t] = tokenizer.pad();
        }
    }
}

