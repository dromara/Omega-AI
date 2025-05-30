package com.omega.example.rnn.data;

import com.omega.common.utils.MathUtils;
import com.omega.engine.tensor.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OneHotDataLoader extends RNNDataLoader {
    public int characters = 1;
    public Map<Character, Integer> dictionary = new HashMap<Character, Integer>();
    public Character[] dictionaryData;
    public int inputType = 0;
    private String dataPath;
    private int dataSize = 0;
    private Character[] data;

    public OneHotDataLoader(String dataPath, int time, int batchSize) {
        this.dataPath = dataPath;
        this.time = time;
        this.loadDataForTXT();
        this.dataSize = data.length;
        this.number = this.dataSize - time;
        this.characters = dictionary.size();
        System.out.println("dataSize[" + dataSize + "] characters[" + characters + "]");
        this.batchSize = batchSize;
    }

    public OneHotDataLoader(String dataPath, int time, int batchSize, int inputType) {
        this.dataPath = dataPath;
        this.time = time;
        this.inputType = inputType;
        this.loadDataForTXT();
        this.dataSize = data.length;
        this.number = this.dataSize - time;
        this.characters = dictionary.size();
        System.out.println("dataSize[" + dataSize + "] characters[" + characters + "]");
        this.batchSize = batchSize;
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
            for (Character key : dictionary.keySet()) {
                dictionaryData[dictionary.get(key)] = key;
            }
            data = new Character[chars.size()];
            data = chars.toArray(data);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            for (int t = 0; t < time; t++) {
                format(i, indexs[i], t, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void format(int b, int i, int t, Tensor input, Tensor label) {
        char curr = data[i + t];
        char next = data[i + t + 1];
        if (inputType == 1) {
            input.data[(t * batchSize + b)] = dictionary.get(curr);
            label.data[(t * batchSize + b) * characters + dictionary.get(next)] = 1.0f;
        } else {
            input.data[(t * batchSize + b) * characters + dictionary.get(curr)] = 1.0f;
            label.data[(t * batchSize + b) * characters + dictionary.get(next)] = 1.0f;
        }
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        this.loadData(indexs, input, label);
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
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

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
    }
}

