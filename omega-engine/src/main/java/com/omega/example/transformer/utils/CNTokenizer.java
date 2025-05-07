package com.omega.example.transformer.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CNTokenizer extends BaseTokenizer {
    public int characters = 1;
    public Map<Character, Integer> dictionary = new HashMap<Character, Integer>();
    public Character[] dictionaryData;
    public String[] vocab;
    public int inputType = 0;
    public int time;
    public int number;
    public int batchSize;
    public Tensor testInput;
    public Character[] trainData;
    public Character[] vailData;
    private String dataPath;
    private int dataSize = 0;
    private Character[] data;
    private float vailRatio = 0.1f;

    public CNTokenizer(String dataPath, int time, int batchSize) {
        this.dataPath = dataPath;
        this.time = time;
        this.loadDataForTXT();
        this.dataSize = data.length;
        this.number = this.dataSize - time;
        this.characters = dictionary.size();
        System.out.println("dataSize[" + dataSize + "] characters[" + characters + "]");
        this.batchSize = batchSize;
        buildData();
    }

    public CNTokenizer(String dataPath, int time, int batchSize, int inputType) {
        this.dataPath = dataPath;
        this.time = time;
        this.inputType = inputType;
        this.loadDataForTXT();
        this.dataSize = data.length;
        this.number = this.dataSize - time;
        this.characters = dictionary.size();
        System.out.println("dataSize[" + dataSize + "] characters[" + characters + "]");
        this.batchSize = batchSize;
        buildData();
    }

    public static Tensor getPositions(int b, int time) {
        float[] data = new float[b * time * time];
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                data[n * time * time + t * time + t] = 1;
            }
        }
        Tensor positions = new Tensor(b * time, 1, 1, time, data, true);
        return positions;
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

    public void buildData() {
        int trainDataSize = (int) (this.number * (1 - vailRatio));
        int vailDataSize = this.number - trainDataSize;
        trainData = new Character[trainDataSize];
        vailData = new Character[vailDataSize];
        System.arraycopy(data, 0, trainData, 0, trainData.length);
        System.arraycopy(data, trainData.length, vailData, 0, vailData.length);
    }

    public void loadDataForTXT() {
        try (FileInputStream fin = new FileInputStream(this.dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            int dic_index = 0;
            String strTmp = "";
            List<Character> chars = new ArrayList<Character>();
            while ((strTmp = buffReader.readLine()) != null) {
                char[] lines = strTmp.toCharArray();
                for (char txt : lines) {
                    chars.add(txt);
                    if (!dictionary.containsKey(txt)) {
                        dictionary.put(txt, dic_index);
                        dic_index++;
                    }
                }
            }
            dictionaryData = new Character[dictionary.size()];
            vocab = new String[dictionary.size()];
            for (Character key : dictionary.keySet()) {
                dictionaryData[dictionary.get(key)] = key;
                vocab[dictionary.get(key)] = key.toString();
            }
            data = new Character[chars.size()];
            data = chars.toArray(data);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            for (int t = 0; t < time; t++) {
                format(i, indexs[i], t, trainData, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadIDXData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            for (int t = 0; t < time; t++) {
                formatIdx(i, indexs[i], t, trainData, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadDataVail(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            for (int t = 0; t < time; t++) {
                format(i, indexs[i], t, vailData, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void format(int b, int i, int t, Character[] dataset, Tensor input, Tensor label) {
        char curr = dataset[i + t];
        char next = dataset[i + t + 1];
        if (inputType == 1) {
            input.data[b * time + t] = dictionary.get(curr);
            label.data[(b * time + t) * characters + dictionary.get(next)] = 1.0f;
        } else {
            input.data[b * time + t] = dictionary.get(curr);
            label.data[(b * time + t) * characters + dictionary.get(next)] = 1.0f;
        }
    }

    public void formatIdx(int b, int i, int t, Character[] dataset, Tensor input, Tensor label) {
        char curr = dataset[i + t];
        char next = dataset[i + t + 1];
        input.data[b * time + t] = dictionary.get(curr);
        label.data[b * time + t] = dictionary.get(next);
    }

    public void format(int b, int i, int t, Tensor input, Tensor label) {
        char curr = data[i + t];
        char next = data[i + t + 1];
        if (inputType == 1) {
            input.data[b * time + t] = dictionary.get(curr);
            label.data[(b * time + t) * characters + dictionary.get(next)] = 1.0f;
        } else {
            input.data[b * time + t] = dictionary.get(curr);
            label.data[(b * time + t) * characters + dictionary.get(next)] = 1.0f;
        }
    }

    public void format(int t, Tensor input, char curr) {
        input.data[t * characters + dictionary.get(curr)] = 1.0f;
    }

    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        this.loadData(indexs, input, label);
    }

    public Tensor initLabelTensor() {
        // TODO Auto-generated method stub
        return new Tensor(time * batchSize, 1, 1, characters, true);
    }

    public int[] getIndexsByAsc(int pageIndex, int batchSize) {
        int start = pageIndex * batchSize;
        int end = pageIndex * batchSize + batchSize;
        if (end > number) {
            start = start - (end - number);
        }
        int[] indexs = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indexs[i] = start + i;
        }
        return indexs;
    }

    public Tensor loadByTxt(String txt) {
        char[] onceToken = txt.toCharArray();
        //		String[] onceToken = txt.split("");
        System.out.println(JsonUtils.toJson(onceToken));
        testInput = Tensor.createTensor(testInput, onceToken.length, 1, 1, characters, true);
        testInput.clear();
        for (int t = 0; t < onceToken.length; t++) {
            format(t, testInput, onceToken[t]);
        }
        testInput.hostToDevice();
        return testInput;
    }
}

