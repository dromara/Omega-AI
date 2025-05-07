package com.omega.example.transformer.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.PrintUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

public class CNChatTokenizer extends BaseTokenizer {
    public static final Map<String, String> specials_dictionary = new HashMap<String, String>() {
        /**
         *

         */
        private static final long serialVersionUID = 342465861011632616L;

        {
            put("<pad>", "#");
            put("<sos>", "@");
            put("<eos>", "-");
            put("<sep>", " ");
        }
    };
    public static final Map<String, String> sd = new HashMap<String, String>() {
        /**
         *

         */
        private static final long serialVersionUID = 3669659616912512613L;

        {
            put("#", "<pad>");
            put("@", "<sos>");
            put("-", "<eos>");
            put(" ", "<sep>");
        }
    };
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+"};
    private final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
    private final String[] specials = new String[]{"<pad>", "<sos>", "<eos>", "<sep>"};
    public int number = 0;
    public Map<String, Integer> dictionary = new HashMap<String, Integer>();
    public List<String[]> tokens = new ArrayList<String[]>();
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    public List<String[]> trainData;
    public List<String[]> vailData;
    private float vailRatio = 0.1f;
    private int batchSize = 1;
    private String dataPath;
    private List<String> org_tokens = new ArrayList<String>();
    private int[] targetLens;

    public CNChatTokenizer(String dataPath, int max_len, int batchSize) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        loadDataForTXT();
        Collections.shuffle(tokens);
        this.number = org_tokens.size();
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
        trainData = tokens.subList(0, trainDataSize);
        vailData = tokens.subList(trainDataSize, trainDataSize + vailDataSize);
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
                    org_tokens.add(strTmp);
                    //	        		strTmp = "<sos>" + strTmp + "<eos>";
                    //		        	System.out.println(strTmp);
                    //		        	System.out.println("line["+dic_index+"]:" + strTmp);
                    //		        	dic_index++;
                }
            }
            buildVocab();
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
            String[] once = org_tokens.get(i).split("");
            //			System.out.println(JsonUtils.toJson(once));
            if (once.length > 1) {
                tokens.add(once);
                for (int j = 0; j < once.length; j++) {
                    String txt = once[j];
                    if (!txt.equals("")) {
                        if (txt.equals(" ")) {
                            txt = "<sep>";
                            once[j] = "<sep>";
                        }
                        if (!dictionary.containsKey(txt)) {
                            dictionary.put(txt, idx);
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
        String[] onceToken = txt.split("");
        System.out.println(JsonUtils.toJson(onceToken));
        testInput = Tensor.createTensor(testInput, max_len, 1, 1, vocab_size, true);
        testInput.clear();
        for (int t = 0; t < max_len; t++) {
            formatNotHeadToIdx(t, onceToken, testInput);
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(String txt) {
        String[] onceToken = txt.split("");
        //		System.out.println(JsonUtils.toJson(onceToken));
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        testInput.clear();
        for (int t = 0; t < txt.length(); t++) {
            formatNotHeadToIdx(t, onceToken, testInput);
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idx) {
        //		System.out.println(JsonUtils.toJson(onceToken));
        testInput = Tensor.createTensor(testInput, idx.length, 1, 1, 1, true);
        testInput.clear();
        for (int t = 0; t < idx.length; t++) {
            formatNotHeadToIdx(t, idx, testInput);
        }
        testInput.hostToDevice();
        return testInput;
    }

    public int[] encode(String txt) {
        String[] onceToken = txt.split("");
        int[] idx = new int[onceToken.length];
        for (int i = 0; i < idx.length; i++) {
            if (onceToken[i].equals(" ")) {
                idx[i] = 3;
            } else if (!dictionary.containsKey(onceToken[i])) {
                idx[i] = 0;
            } else {
                idx[i] = dictionary.get(onceToken[i]);
            }
        }
        return idx;
    }

    public String decode(int[] idx) {
        String txt = "";
        for (int i = 0; i < idx.length; i++) {
            txt += vocab[idx[i]];
        }
        return txt;
    }

    public String decode(int[] idx, int skip) {
        String txt = "";
        for (int i = skip; i < idx.length; i++) {
            txt += vocab[idx[i]];
        }
        return txt;
    }

    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = tokens.get(indexs[i]);
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

    public void loadTrainData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = trainData.get(indexs[i]);
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

    public void loadTrainIdx(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = trainData.get(indexs[i]);
            //			System.out.println(onceToken.length);
            for (int t = 0; t < max_len; t++) {
                formatNotHead2Idx(i, t, onceToken, input, label);
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
            String[] onceToken = vailData.get(indexs[i]);
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

    public void loadData(int[] indexs, Tensor input, Tensor label, Tensor mask) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            String[] onceToken = tokens.get(indexs[i]);
            getTargetLens()[i] = onceToken.length;
            //			System.out.println(onceToken.length);
            for (int t = 0; t < max_len; t++) {
                format(i, t, onceToken, input, label);
            }
        }
        triu(1, getTargetLens(), mask);
        //		System.out.println(JsonUtils.toJson(getTargetLens()));
        //		mask.showDM();
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void formatOnce(int t, String[] onceToken, Tensor input) {
        if (t == 0) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            input.data[t * vocab_size + 1] = 1.0f;
            input.data[(t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
            return;
        }
        if (t == onceToken.length - 1) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            System.out.println("curr:" + curr);
            input.data[(t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
            return;
        }
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            System.out.println("curr:" + curr);
            input.data[(t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
        } else if (t < max_len - 1) {
            input.data[(t + 1) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatOnceNotHead(int t, String[] onceToken, Tensor input) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            input.data[t * vocab_size + dictionary.get(curr)] = 1.0f;
        } else if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            input.data[t * vocab_size + dictionary.get(curr)] = 1.0f;
        } else {
            input.data[t * vocab_size + 0] = 1.0f;
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
    public void formatNotHeadToIdx(int t, String[] onceToken, Tensor input) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            input.data[t] = dictionary.get(curr);
        } else if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            if (sd.get(curr) != null) {
                curr = sd.get(curr);
            }
            input.data[t] = dictionary.get(curr);
        } else {
            input.data[t] = dictionary.get("<pad>");
        }
    }

    public void formatNotHeadToIdx(int t, int[] onceToken, Tensor input) {
        if ((t + 1) < onceToken.length) {
            input.data[t] = onceToken[t];
        } else if ((t + 1) == onceToken.length) {
            input.data[t] = onceToken[t];
        } else {
            input.data[t] = dictionary.get("<pad>");
        }
    }

    public void format(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if (t == 0) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[(b * max_len + t) * vocab_size + 1] = 1.0f;
            input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
            return;
        }
        if (t == onceToken.length - 1) {
            String curr = onceToken[t];
            input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 2] = 1.0f;
            return;
        }
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            //			System.out.println(next);
            input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
        } else {
            input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatNotHead(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[(b * max_len + t) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
        } else if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            input.data[(b * max_len + t) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get("<sep>")] = 1.0f;
        } else {
            input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatNotHeadToIdx(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[b * max_len + t] = dictionary.get(curr);
            label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
        } else if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            input.data[b * max_len + t] = dictionary.get(curr);
            label.data[(b * max_len + t) * vocab_size + dictionary.get("<sep>")] = 1.0f;
        } else {
            input.data[b * max_len + t] = dictionary.get("<pad>");
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatNotHead2Idx(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) < onceToken.length) {
            String curr = onceToken[t];
            String next = onceToken[t + 1];
            input.data[b * max_len + t] = dictionary.get(curr);
            label.data[b * max_len + t] = dictionary.get(next);
        } else if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            input.data[b * max_len + t] = dictionary.get(curr);
            label.data[b * max_len + t] = dictionary.get("<sep>");
        } else {
            input.data[b * max_len + t] = dictionary.get("<pad>");
            label.data[b * max_len + t] = dictionary.get("<pad>");
        }
    }

    public int[] getTargetLens() {
        if (targetLens == null || targetLens.length != batchSize) {
            targetLens = new int[batchSize];
        }
        return targetLens;
    }
}

