package com.omega.example.transformer.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.engine.tensor.Tensor;

public class ENTokenizer {
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+"};
    private final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
    private final String[] specials = new String[]{"<pad>", "<sos>", "<eos>"};
    public int number = 0;
    public Map<String, Integer> dictionary = new HashMap<String, Integer>();
    public List<String[]> tokens = new ArrayList<String[]>();
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    public int formatType = 0; //0:word,1:char
    private int batchSize = 1;
    private String dataPath;
    private List<String> org_tokens = new ArrayList<String>();

    public ENTokenizer(String dataPath, int max_len, int batchSize) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        loadDataForTXT();
        this.number = org_tokens.size();
        System.out.println(this.number);
    }

    public ENTokenizer(String dataPath, int max_len, int batchSize, int formatType) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.formatType = formatType;
        if (formatType > 0) {
            loadDataForTXTByChar();
        } else {
            loadDataForTXT();
        }
        this.number = org_tokens.size();
        System.out.println(this.number);
    }

    public static Tensor getPositions(int b, int time) {
        float[] data = new float[b * time * time];
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                data[n * time * time + t * time + t] = 1;
            }
        }
        Tensor positions = new Tensor(b * time, 1, 1, time, data, true);
        //		positions.showDMByNumber(0);
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

    public static void main(String[] args) {
        String dataPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";
        int batchSize = 64;
        ENTokenizer tokenizer = new ENTokenizer(dataPath, 256, batchSize);
        tokenizer.loadDataForTXT();
        Tensor subsequent_mask = tokenizer.triu(2, 4, 5, 5, 1);
        subsequent_mask.showDM();
        Tensor positions = getPositions(2, 4);
        positions.showDM();
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
                if (!strTmp.equals(" ")) {
                    strTmp = "<sos>" + strTmp + "<eos>";
                    org_tokens.add(strTmp);
                    //		        	System.out.println("" + strTmp);
                    //		        	dic_index++;
                }
            }
            buildVocab();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataForTXTByChar() {
        try (FileInputStream fin = new FileInputStream(this.dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            //			int dic_index = 0;
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                for (int i = 0; i < _patterns.length; i++) {
                    strTmp = strTmp.replaceAll(_patterns[i], _replacements[i]);
                }
                strTmp = strTmp.toLowerCase();
                if (!strTmp.equals(" ")) {
                    strTmp = "<sos>" + strTmp + "<eos>";
                    org_tokens.add(strTmp);
                    //		        	System.out.println("" + strTmp);
                    //		        	dic_index++;
                }
            }
            buildVocabByChar();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void buildVocab() {
        for (int i = 0; i < specials.length; i++) {
            dictionary.put(specials[i], i);
        }
        int idx = specials.length;
        for (int i = 0; i < org_tokens.size(); i++) {
            String[] once = org_tokens.get(i).split(" ");
            if (once.length > 1) {
                tokens.add(once);
                for (int j = 0; j < once.length; j++) {
                    if (!once[j].equals("")) {
                        if (!dictionary.containsKey(once[j])) {
                            dictionary.put(once[j], idx);
                            idx++;
                        }
                    }
                }
            }
        }
        vocab_size = dictionary.size();
        vocab = new String[vocab_size];
        for (String key : dictionary.keySet()) {
            vocab[dictionary.get(key)] = key;
        }
    }

    public void buildVocabByChar() {
        for (int i = 0; i < specials.length; i++) {
            dictionary.put(specials[i], i);
        }
        int idx = specials.length;
        for (int i = 0; i < org_tokens.size(); i++) {
            String[] once = org_tokens.get(i).split("");
            if (once.length > 1) {
                tokens.add(once);
                for (int j = 0; j < once.length; j++) {
                    if (!once[j].equals("")) {
                        if (!dictionary.containsKey(once[j])) {
                            dictionary.put(once[j], idx);
                            idx++;
                        }
                    }
                }
            }
        }
        vocab_size = dictionary.size();
        vocab = new String[vocab_size];
        for (String key : dictionary.keySet()) {
            vocab[dictionary.get(key)] = key;
        }
    }

    public Tensor loadByTxt(String txt) {
        char[] en_lines = txt.toLowerCase().toCharArray();
        testInput = Tensor.createTensor(testInput, max_len, 1, 1, vocab_size, true);
        testInput.clear();
        for (int t = 0; t < max_len; t++) {
            int valIdx = dictionary.get("<PAD>");
            if (t < en_lines.length) {
                valIdx = dictionary.get(en_lines[t] + "");
            }
            testInput.data[t * vocab_size + valIdx] = 1.0f;
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = tokens.get(indexs[i]);
            //			System.out.println(JsonUtils.toJson(onceToken));
            //			System.out.println(onceToken.length);
            for (int t = 0; t < max_len; t++) {
                format(i, t, onceToken, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadDataIdx(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = tokens.get(indexs[i]);
            for (int t = 0; t < max_len; t++) {
                formatIdx(i, t, onceToken, input, label);
            }
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void format(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[(b * max_len + t) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
        } else {
            input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatIdx(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[b * max_len + t] = dictionary.get(curr);
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
        } else {
            input.data[b * max_len + t] = dictionary.get("<pad>");
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }
}

