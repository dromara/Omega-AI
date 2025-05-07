package com.omega.example.rnn.data;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MathUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class IndexDataLoader extends RNNDataLoader {
    public int en_characters = 1;
    public int ch_characters = 1;
    public Map<String, Integer> en_dictionary = new HashMap<String, Integer>();
    public Map<String, Integer> ch_dictionary = new HashMap<String, Integer>();
    public String[] ch_dic;
    public String[] en_dic;
    public int max_en = 0;
    public int max_ch = 0;
    private String dataPath;
    private int dataSize = 0;
    private List<char[]> en_chars = new ArrayList<char[]>();
    private List<char[]> ch_chars = new ArrayList<char[]>();
    private Tensor testInput;

    public IndexDataLoader(String dataPath, int batchSize) {
        this.dataPath = dataPath;
        this.loadDataForCSV();
        this.dataSize = en_chars.size();
        this.number = this.dataSize - time;
        System.out.println("dataSize[" + dataSize + "] ch_characters[" + ch_characters + "] en_characters[" + en_characters + "]");
        this.batchSize = batchSize;
    }

    public void loadDataForCSV() {
        try (FileInputStream fin = new FileInputStream(this.dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            int en_dic_index = 0;
            int ch_dic_index = 0;
            int idx = 0;
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    char[] en_lines = list[0].toLowerCase().toCharArray();
                    char[] ch_lines = list[1].toCharArray();
                    en_chars.add(en_lines);
                    ch_chars.add(ch_lines);
                    if (en_lines.length >= max_en) {
                        max_en = en_lines.length;
                    }
                    if (ch_lines.length >= max_ch) {
                        max_ch = ch_lines.length;
                    }
                    for (char txt : en_lines) {
                        if (!en_dictionary.containsKey(txt + "")) {
                            en_dictionary.put(txt + "", en_dic_index);
                            en_dic_index++;
                        }
                    }
                    for (char txt : ch_lines) {
                        if (!ch_dictionary.containsKey(txt + "")) {
                            ch_dictionary.put(txt + "", ch_dic_index);
                            ch_dic_index++;
                        }
                    }
                }
                idx++;
            }
            en_dictionary.put("<PAD>", en_dic_index++);
            ch_dictionary.put("<PAD>", ch_dic_index++);
            ch_dictionary.put("<BOS>", ch_dic_index++);
            ch_dictionary.put("<EOS>", ch_dic_index++);
            this.en_characters = en_dictionary.size();
            this.ch_characters = ch_dictionary.size();
            en_dic = new String[en_characters];
            ch_dic = new String[ch_characters];
            for (String key : en_dictionary.keySet()) {
                en_dic[en_dictionary.get(key)] = key;
            }
            for (String key : ch_dictionary.keySet()) {
                ch_dic[ch_dictionary.get(key)] = key;
            }
            max_ch += 2;
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
    }

    public Tensor loadByTxt(String txt) {
        char[] en_lines = txt.toLowerCase().toCharArray();
        testInput = Tensor.createTensor(testInput, max_en, 1, 1, en_characters, true);
        testInput.clear();
        for (int t = 0; t < max_en; t++) {
            int valIdx = en_dictionary.get("<PAD>");
            if (t < en_lines.length) {
                valIdx = en_dictionary.get(en_lines[t] + "");
            }
            testInput.data[t * en_characters + valIdx] = 1.0f;
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void loadData(int[] indexs, Tensor enInput, Tensor deInput, Tensor label) {
        enInput.clear();
        deInput.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            format(i, indexs[i], enInput, deInput, label);
        }
        /**
         * copy data to gpu.

         */
        enInput.hostToDevice();
        deInput.hostToDevice();
        label.hostToDevice();
    }

    public void format(int b, int idx, Tensor enInput, Tensor deInput, Tensor label) {
        /**
         * 装载编码器数据

         */
        char[] en_txt = en_chars.get(idx);
        for (int t = 0; t < max_en; t++) {
            int valIdx = en_dictionary.get("<PAD>");
            if (t < en_txt.length) {
                valIdx = en_dictionary.get(en_txt[t] + "");
            }
            enInput.data[t * batchSize * en_characters + b * en_characters + valIdx] = 1.0f;
        }
        /**
         * 装载解码器数据

         */
        char[] ch_txt = ch_chars.get(idx);
        for (int t = 0; t < max_ch - 1; t++) {
            int valIdx = ch_dictionary.get("<PAD>");
            if (t == 0) {
                valIdx = ch_dictionary.get("<BOS>");
            } else if ((t - 1) < ch_txt.length) {
                valIdx = ch_dictionary.get(ch_txt[t - 1] + "");
            }
            deInput.data[t * batchSize * ch_characters + b * ch_characters + valIdx] = 1.0f;
        }
        for (int t = 0; t < max_ch - 1; t++) {
            int valIdx = ch_dictionary.get("<PAD>");
            if (t < ch_txt.length) {
                valIdx = ch_dictionary.get(ch_txt[t] + "");
            } else if (t == ch_txt.length) {
                valIdx = ch_dictionary.get("<EOS>");
            }
            label.data[t * batchSize * ch_characters + b * ch_characters + valIdx] = 1.0f;
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
        return new Tensor(batchSize, 1, 1, max_ch, true);
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

