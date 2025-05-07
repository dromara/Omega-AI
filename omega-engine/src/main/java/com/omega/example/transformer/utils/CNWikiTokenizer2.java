package com.omega.example.transformer.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.PrintUtils;
import com.omega.example.transformer.tokenizer.bertTokenizer.BertTokenizer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class CNWikiTokenizer2 extends BaseTokenizer {
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
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+", "\\r", "\n"};
    private final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " ", "", ""};
    private final String[] specials = new String[]{"<pad>", "<sos>", "<eos>", "<sep>"};
    public int number = 0;
    public Map<String, Integer> dictionary = new HashMap<String, Integer>();
    //	private List<String> org_tokens = new ArrayList<String>();
    //	public List<String[]> tokens = new ArrayList<String[]>();
    public BertTokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    public List<Integer[]> trainData;
    public List<Integer[]> vailData;
    private float vailRatio = 0.1f;
    private int batchSize = 1;
    private String dataPath;
    private List<Integer[]> idxData = new ArrayList<Integer[]>();
    private int[] targetLens;
    private int dic_idx = 0;
    private int maxDataCount = 0;
    private int min_len = 30;

    public CNWikiTokenizer2(String dataPath, int max_len, int batchSize) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        loadDataForJson();
        Collections.shuffle(idxData);
        this.number = idxData.size();
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + vocab_size);
        buildData();
    }

    public CNWikiTokenizer2(String dataPath, int max_len, int batchSize, boolean isBigData, int maxDataCount) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.maxDataCount = maxDataCount;
        if (isBigData) {
            loadLargeJsonData();
        } else {
            loadDataForJson();
        }
        Collections.shuffle(idxData);
        this.number = idxData.size();
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + vocab_size);
        buildData();
    }

    public CNWikiTokenizer2(String dataPath, int max_len, int batchSize, BertTokenizer tokenizer, int maxDataCount) {
        this.dataPath = dataPath;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.vocab_size();
        this.maxDataCount = maxDataCount;
        loadTokenizerData();
        //		if(isBigData) {
        //			loadLargeJsonData();
        //		}else {
        //			loadDataForJson();
        //		}
        //		Collections.shuffle(idxData);
        this.number = idxData.size();
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
        trainData = idxData.subList(0, trainDataSize);
        vailData = idxData.subList(trainDataSize, trainDataSize + vailDataSize);
        idxData = null;
    }

    public void loadTokenizerData() {
        try (FileReader fileReader = new FileReader(this.dataPath); BufferedReader bufferedReader = new BufferedReader(fileReader)) {
            String line;
            int index = 0;
            while ((line = bufferedReader.readLine()) != null) {
                //			    	System.out.println(line);
                String[] ids = line.split(" ");
                if (ids.length >= min_len) {
                    Integer[] tokens = null;
                    if (ids.length > max_len) {
                        tokens = new Integer[max_len];
                    } else {
                        tokens = new Integer[ids.length];
                    }
                    for (int i = 0; i < tokens.length; i++) {
                        tokens[i] = Integer.parseInt(ids[i]);
                    }
                    idxData.add(tokens);
                }
                if (idxData.size() >= this.maxDataCount) {
                    break;
                }
                index++;
                //				System.out.println(index);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public String decode(Tensor output) {
        int[] tokens = new int[output.number];
        for (int t = 0; t < output.number; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
            tokens[t] = predictIndex;
        }
        return tokenizer.decode(tokens);
    }

    public void loadLargeJsonData() {
        try (FileReader fileReader = new FileReader(this.dataPath); BufferedReader bufferedReader = new BufferedReader(fileReader)) {
            buildCommonVocab();
            String line;
            Map<String, Object> once = new HashMap<String, Object>();
            int index = 0;
            String strTmp = "";
            dic_idx = dictionary.size() - 1;
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                List<Map<String, Object>> sections = (List<Map<String, Object>>) once.get("sections");
                if (once.get("summary") != null && !once.get("summary").toString().equals("")) {
                    strTmp = once.get("title").toString() + "： " + once.get("summary").toString();
                } else {
                    if (sections.size() > 0) {
                        strTmp = once.get("title").toString();
                    }
                }
                for (Map<String, Object> os : sections) {
                    String content = os.get("content").toString();
                    strTmp += os.get("title").toString() + "：" + content + "。";
                }
                for (int p = 0; p < _patterns.length; p++) {
                    strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
                }
                if (!strTmp.equals(" ") && !strTmp.equals("") && strTmp.length() >= min_len) {
                    strTmp.replaceAll(" ", "");
                    if (strTmp.length() > max_len) {
                        strTmp = strTmp.substring(0, max_len);
                    }
                    buildVocab(strTmp);
                }
                if (idxData.size() >= this.maxDataCount) {
                    break;
                }
                index++;
                System.out.println(index);
            }
            buildVocab();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataForJson() {
        buildCommonVocab();
        List<Map<String, String>> list = LagJsonReader.readJsonFileSamll(this.dataPath);
        String strTmp = "";
        dic_idx = dictionary.size() - 1;
        for (int i = 0; i < list.size(); i++) {
            strTmp = list.get(i).get("completion");
            for (int p = 0; p < _patterns.length; p++) {
                strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
            }
            if (!strTmp.equals(" ") && !strTmp.equals("") && strTmp.length() <= max_len - 2) {
                buildVocab(strTmp);
            }
        }
        buildVocab();
    }

    public void buildCommonVocab() {
        for (int i = 0; i < specials.length; i++) {
            dictionary.put(specials[i], i);
        }
    }

    public void buildVocab(String text) {
        String[] once = text.split("");
        if (once.length > 5) {
            Integer[] idx = new Integer[once.length];
            for (int j = 0; j < once.length; j++) {
                String txt = once[j];
                if (!dictionary.containsKey(txt)) {
                    dictionary.put(txt, dic_idx);
                    dic_idx++;
                    //        			System.out.println("dic_idx:"+dic_idx);
                }
                idx[j] = dictionary.get(txt);
            }
            idxData.add(idx);
        }
    }

    public void buildVocab() {
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
        if (tokenizer != null) {
            int[] idx = tokenizer.encode(txt);
            testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
            for (int t = 0; t < txt.length(); t++) {
                testInput.data[t] = idx[t];
            }
        } else {
            String[] onceToken = txt.split("");
            //			System.out.println(JsonUtils.toJson(onceToken));
            testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
            //			testInput.clear();
            for (int t = 0; t < txt.length(); t++) {
                testInput.data[t] = dictionary.get(onceToken[t]);
            }
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(String txt, int maxLen) {
        if (tokenizer != null) {
            int[] idx = tokenizer.encode(txt);
            testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
            for (int t = 0; t < idx.length; t++) {
                testInput.data[t] = idx[t];
            }
        } else {
            String[] onceToken = txt.split("");
            //			System.out.println(JsonUtils.toJson(onceToken));
            testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
            //			testInput.clear();
            for (int t = 0; t < txt.length(); t++) {
                testInput.data[t] = dictionary.get(onceToken[t]);
            }
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs, int maxLen) {
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
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

    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            Integer[] onceToken = idxData.get(indexs[i]);
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
        //		input.clear();
        label.clear();
        for (int i = 0; i < indexs.length; i++) {
            Integer[] onceToken = trainData.get(indexs[i]);
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
            Integer[] onceToken = vailData.get(indexs[i]);
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
            Integer[] onceToken = idxData.get(indexs[i]);
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

    //	public static void getPositions(int b,int time,Tensor positions) {
    //		positions = Tensor.createTensor(positions, b * time, 1, 1, time, true);
    //		for(int n = 0;n<b;n++) {
    //			for(int t = 0;t<time;t++) {
    //				positions.data[n * time * time + t * time + t] = 1;
    //			}
    //		}
    //		positions.hostToDevice();
    //	}
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

    public void format(int b, int t, Integer[] onceToken, Tensor input, Tensor label) {
        if (t == 0) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input.data[(b * max_len + t) * vocab_size + 1] = 1.0f;
            input.data[(b * max_len + t + 1) * vocab_size + curr] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
            return;
        }
        if (t == onceToken.length - 1) {
            int curr = onceToken[t];
            input.data[(b * max_len + t + 1) * vocab_size + curr] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 2] = 1.0f;
            return;
        }
        if ((t + 1) < onceToken.length) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            //			System.out.println(next);
            input.data[(b * max_len + t + 1) * vocab_size + curr] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
        } else {
            input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatNotHead(int b, int t, String[] onceToken, Tensor input, Tensor label) {
        if ((t + 1) == onceToken.length) {
            String curr = onceToken[t];
            input.data[(b * max_len + t) * vocab_size + dictionary.get(curr)] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + dictionary.get("<sep>")] = 1.0f;
        } else {
            input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
        }
    }

    public void formatNotHeadToIdx(int b, int t, Integer[] onceToken, Tensor input, Tensor label) {
        if (tokenizer != null) {
            if (t + 1 < onceToken.length) {
                int curr = onceToken[t];
                int next = onceToken[t + 1];
                input.data[b * max_len + t] = curr;
                label.data[b * max_len + t] = next;
                //				label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
            } else if (t + 1 == onceToken.length) {
                int curr = onceToken[t];
                input.data[b * max_len + t] = curr;
                label.data[b * max_len + t] = tokenizer.eos;
                //				label.data[(b * max_len + t) * vocab_size + tokenizer.eos] = 1.0f;
            } else {
                input.data[b * max_len + t] = tokenizer.pad;
                label.data[b * max_len + t] = tokenizer.pad;
                //				label.data[(b * max_len + t) * vocab_size + tokenizer.pad] = 1.0f;
            }
        } else {
            if (t + 1 < onceToken.length) {
                int curr = onceToken[t];
                int next = onceToken[t + 1];
                input.data[b * max_len + t] = curr;
                label.data[b * max_len + t] = next;
                //				label.data[(b * max_len + t) * vocab_size + next] = 1.0f;
            } else if (t + 1 == onceToken.length) {
                int curr = onceToken[t];
                input.data[b * max_len + t] = curr;
                label.data[b * max_len + t] = dictionary.get("<eos>");
                //				label.data[(b * max_len + t) * vocab_size + dictionary.get("<eos>")] = 1.0f;
            } else {
                input.data[b * max_len + t] = dictionary.get("<pad>");
                label.data[b * max_len + t] = dictionary.get("<pad>");
                //				label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
            }
        }
    }

    public int[] getTargetLens() {
        if (targetLens == null || targetLens.length != batchSize) {
            targetLens = new int[batchSize];
        }
        return targetLens;
    }
}

