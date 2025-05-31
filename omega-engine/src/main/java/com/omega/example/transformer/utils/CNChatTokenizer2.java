package com.omega.example.transformer.utils;

import com.omega.common.utils.PrintUtils;
import com.omega.engine.tensor.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

public class CNChatTokenizer2 extends BaseTokenizer {
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+"};
    private final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
    public BPETokenizer tokenizer;
    public int number = 0;
    public int max_len = 256;
    public int vocab_size;
    public Tensor testInput;
    public List<List<Integer>> trainData;
    public List<List<Integer>> vailData;
    private float vailRatio = 0.1f;
    private int batchSize = 1;
    private String dataPath;
    private List<List<Integer>> org_ids = Collections.synchronizedList(new ArrayList<List<Integer>>());
    private int[] targetLens;

    public CNChatTokenizer2(String dataPath, int max_len, int batchSize, BPETokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        loadDataForTXTSteam();
        Collections.shuffle(org_ids);
        this.number = org_ids.size();
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + vocab_size);
        buildData();
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

    public void buildData() {
        int trainDataSize = (int) (this.number * (1 - vailRatio));
        int vailDataSize = this.number - trainDataSize;
        trainData = org_ids.subList(0, trainDataSize);
        vailData = org_ids.subList(trainDataSize, trainDataSize + vailDataSize);
    }

    public void loadDataForTXTSteam() {
        try {
            Path path = Paths.get(this.dataPath);
            Stream<String> lines = Files.lines(path).parallel();
            lines.forEach(Consumer -> {
                for (int i = 0; i < _patterns.length; i++) {
                    Consumer = Consumer.replaceAll(_patterns[i], _replacements[i]);
                }
                Consumer = Consumer.toLowerCase();
                Consumer = Consumer.substring(0, Consumer.length() - 1);
                if (!Consumer.equals(" ") && !Consumer.equals("") && Consumer.length() <= max_len - 2) {
                    List<Integer> ids = tokenizer.encode(Consumer);
                    if (ids.size() <= max_len - 2) {
                        org_ids.add(ids);
                    }
                }
            });
            System.out.println("[" + org_ids.size() + "]load data finish.");
            buildVocab();
            System.out.println("build vocab finish.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataForTXT() {
        try (FileInputStream fin = new FileInputStream(this.dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            //			int dic_index = 0;
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                for (int i = 0; i < _patterns.length; i++) {
                    strTmp = strTmp.replaceAll(_patterns[i], _replacements[i]);
                }
                strTmp = strTmp.toLowerCase();
                strTmp = strTmp.substring(0, strTmp.length() - 1);
                if (!strTmp.equals(" ") && !strTmp.equals("") && strTmp.length() <= max_len - 2) {
                    List<Integer> ids = tokenizer.encode(strTmp);
                    if (ids.size() <= max_len - 2) {
                        org_ids.add(ids);
                    }
                }
            }
            System.out.println("load data finish.");
            buildVocab();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void buildVocab() {
        tokenizer.addSpecial("<sep>");
        tokenizer.addSpecial("<pad>");
        vocab_size = tokenizer.vocab.size();
    }

    public Tensor loadByTxtToIdx(String txt) {
        List<Integer> onceToken = tokenizer.encode(txt);
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        testInput.clear();
        for (int t = 0; t < txt.length(); t++) {
            formatNotHeadToIdx(t, onceToken, testInput);
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void loadTrainData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            List<Integer> onceToken = trainData.get(indexs[i]);
            //			System.out.println(onceToken.length);
            for (int t = 0; t < max_len; t++) {
                formatNotHeadToIdx(i, t, onceToken, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadVailData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            List<Integer> onceToken = vailData.get(indexs[i]);
            //			System.out.println(onceToken.length);
            for (int t = 0; t < max_len; t++) {
                formatNotHeadToIdx(i, t, onceToken, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void formatNotHeadToIdx(int t, List<Integer> onceToken, Tensor input) {
        if ((t + 1) < onceToken.size()) {
            input.data[t] = onceToken.get(t);
        } else {
            input.data[t] = tokenizer.specials.get("<pad>");
        }
    }

    public void formatNotHeadToIdx(int b, int t, List<Integer> onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.size()) {
            input.data[b * max_len + t] = onceToken.get(t);
            if ((t + 2) < onceToken.size()) {
                label.data[(b * max_len + t) * vocab_size + onceToken.get(t + 2)] = 1.0f;
            } else {
                //				label.data[(b * max_len + t) * vocab_size + onceToken.get(t + 1)] = 1.0f;
                label.data[(b * max_len + t) * vocab_size + tokenizer.specials.get("<sep>")] = 1.0f;
            }
            //			label.data[(b * max_len + t) * vocab_size + onceToken.get(t + 1)] = 1.0f;
            //			label.data[(b * max_len + t) * vocab_size + onceToken.get(t + 2)] = 1.0f;
        } else if ((t + 1) == onceToken.size()) {
            input.data[b * max_len + t] = onceToken.get(t);
            label.data[(b * max_len + t) * vocab_size + tokenizer.specials.get("<sep>")] = 1.0f;
        } else {
            input.data[b * max_len + t] = tokenizer.specials.get("<pad>");
            label.data[(b * max_len + t) * vocab_size + tokenizer.specials.get("<pad>")] = 1.0f;
        }
    }

    public int[] getTargetLens() {
        if (targetLens == null || targetLens.length != batchSize) {
            targetLens = new int[batchSize];
        }
        return targetLens;
    }
}

