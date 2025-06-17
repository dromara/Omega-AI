package com.omega.example.transformer.utils.bpe;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * bpe tokenizer
 *
 * @author Administrator
 */
public class BPETokenizer3 extends Tokenizer {
    public Map<String, Integer> vocab;
    public Map<Integer, String> decoder = new HashMap<Integer, String>();
    public Map<String[], Integer> merges;
    public Map<Integer, String> unicodeMap;
    public int voc_size = 0;
    public int pad = 0;
    public int sos = 1;
    public int eos = 2;

    public BPETokenizer3(String vocabPath, String mergesPath) {
        System.out.println("init bpe tokenizer.");
        this.unicodeMap = unicodeMap();
        this.vocab = readJsonFileSamll(vocabPath);
        this.voc_size = vocab.size();
        this.merges = readMerges(mergesPath);
        for (String key : vocab.keySet()) {
            decoder.put(vocab.get(key).intValue(), key);
        }
    }

    public static String jb2pb(int b) {
        if (b < 0) {
            b = b + 256;
        }
        return b + "";
    }

    public static int jb2pbToInt(int b) {
        if (b < 0) {
            b = b + 256;
        }
        return b;
    }

    public static int[] jb2pbToInt(byte[] b) {
        int[] v = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            if (b[i] < 0) {
                v[i] = b[i] + 256;
            } else {
                v[i] = b[i];
            }
        }
        return v;
    }

    public static byte pb2jb(int b) {
        if (b > 127) {
            b = b - 256;
        }
        return (byte) b;
    }

    public static String[] mergeToken(String[] chars, int index, String pair) {
        String[] result = new String[chars.length - 1];
        int offset = 0;
        for (int i = 0; i < result.length; i++) {
            if (i == index) {
                result[i] = pair;
                offset++;
            } else {
                result[i] = chars[i + offset];
            }
        }
        return result;
    }

    public static Map<Integer, String> unicodeMap() {
        int start1 = ord('!');
        int end1 = ord('~') + 1;
        Integer[] one = MatrixUtils.orderRankInt(start1, end1);
        //		System.out.println(JsonUtils.toJson(one));
        int start2 = ord('¡');
        int end2 = ord('¬') + 1;
        Integer[] two = MatrixUtils.orderRankInt(start2, end2);
        //		System.out.println(JsonUtils.toJson(two));
        int start3 = ord('®');
        int end3 = ord('ÿ') + 1;
        Integer[] three = MatrixUtils.orderRankInt(start3, end3);
        //		System.out.println(JsonUtils.toJson(three));
        Integer[] result = new Integer[one.length + two.length + three.length];
        System.arraycopy(one, 0, result, 0, one.length);
        System.arraycopy(two, 0, result, one.length, two.length);
        System.arraycopy(three, 0, result, one.length + two.length, three.length);
        List<Integer> bs = new ArrayList<Integer>(Arrays.asList(result));
        List<Integer> cs = new ArrayList<Integer>(Arrays.asList(result));
        //		System.out.println(JsonUtils.toJson(result));
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }
        //		System.out.println(JsonUtils.toJson(cs));
        List<Character> final_cs = new ArrayList<Character>();
        for (int i = 0; i < cs.size(); i++) {
            //			System.out.println(cs.get(i).intValue());
            char c = (char) cs.get(i).intValue();
            //			System.out.println(c);
            final_cs.add(c);
        }
        //		System.out.println(final_cs.size());
        //		System.out.println(JsonUtils.toJson(final_cs));
        Map<Integer, String> unicodeMap = new HashMap<Integer, String>();
        for (int i = 0; i < final_cs.size(); i++) {
            unicodeMap.put(bs.get(i), final_cs.get(i).toString());
        }
        return unicodeMap;
    }

    public static int ord(char ch) {
        int asciiValue = ch;
        return asciiValue;
    }

    public static String unicodeToken(int[] tokenIds, Map<Integer, String> unicodeMap) {
        String txt = "";
        for (int idx : tokenIds) {
            txt += unicodeMap.get(idx);
        }
        return txt;
    }

    // 读取json文件并解析为对象
    public static void main(String args[]) {
        try {
            String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
            String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
            String txt = "<s>user你</s>";
            BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
            List<Integer> ids = bpe.encode(txt);
            System.out.println(JsonUtils.toJson(ids));
            String decodeTxt = bpe.decode(ids);
            System.out.println(decodeTxt);
            String txt2 = "<s>请问你在哪里？fsdfsdfsdfsdfdsfsd</s>";
            System.out.println(JsonUtils.toJson(bpe.encodeInt(txt2, 512)));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public String decode(List<Integer> ids) {
        String txt = "";
        for (int i = 0; i < ids.size(); i++) {
            String bstr = decoder.get(ids.get(i));
            txt += bstr;
        }
        return new String(unicode2Byte(txt, unicodeMap), StandardCharsets.UTF_8);
    }

    @Override
    public String decode(int[] ids) {
        String txt = "";
        for (int i = 0; i < ids.length; i++) {
            String bstr = decoder.get(ids[i]);
            txt += bstr;
        }
        return new String(unicode2Byte(txt, unicodeMap), StandardCharsets.UTF_8);
    }

    private byte[] unicode2Byte(String txt, Map<Integer, String> unicodeMap) {
        String[] tokens = txt.split("");
        byte[] y = new byte[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            y[i] = pb2jb(getKeyByValue(token, unicodeMap));
        }
        return y;
    }

    private Integer getKeyByValue(String val, Map<Integer, String> unicodeMap) {
        for (Integer key : unicodeMap.keySet()) {
            if (unicodeMap.get(key).equals(val)) {
                return key;
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Integer> readJsonFileSamll(String path) {
        Map<String, Double> mapListd = new HashMap<String, Double>();
        Map<String, Integer> mapList = new HashMap<String, Integer>();
        try {
            String line = null;
            FileInputStream fis = new FileInputStream(path);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            while ((line = bufferedReader.readLine()) != null) {
                System.out.println(line);
                mapListd = JsonUtils.gson.fromJson(line, mapList.getClass());
            }
            bufferedReader.close();
            for (String key : mapListd.keySet()) {
                mapList.put(key, mapListd.get(key).intValue());
            }
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private Map<String[], Integer> readMerges(String path) {
        Map<String[], Integer> mapList = new LinkedHashMap<String[], Integer>();
        try {
            String line = null;
            FileInputStream fis = new FileInputStream(path);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int index = 0;
            while ((line = bufferedReader.readLine()) != null) {
                //				System.out.println(line);
                mapList.put(line.split(" "), index);
                index++;
            }
            bufferedReader.close();
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public List<Integer> encode(String txt) {
        String unicodeToken = unicodeToken(encode(txt, "utf-8"), unicodeMap);
        //		System.out.println("index2:"+i);
        //		System.out.println(unicodeToken);
        //
        String[] bbpeTokens = bbpe(unicodeToken, merges);
        //		System.out.println("index3:"+i);
        //		System.out.println(JsonUtils.toJson(bbpeTokens));
        List<Integer> idxs = new ArrayList<Integer>();
        for (String key : bbpeTokens) {
            idxs.add(vocab.get(key).intValue());
        }
        //		System.out.println("index4:"+i);
        return idxs;
    }

    @Override
    public int[] encodeInt(String txt) {
        String unicodeToken = unicodeToken(encode(txt, "utf-8"), unicodeMap);
        String[] bbpeTokens = bbpe(unicodeToken, merges);
        int[] idxs = new int[bbpeTokens.length];
        for (int i = 0; i < bbpeTokens.length; i++) {
            String key = bbpeTokens[i];
            idxs[i] = vocab.get(key).intValue();
        }
        return idxs;
    }

    @Override
    public int[] encodeInt(String txt, int maxLen) {
        String unicodeToken = unicodeToken(encode(txt, "utf-8"), unicodeMap);
        String[] bbpeTokens = bbpe(unicodeToken, merges);
        int[] idxs = new int[maxLen];
        for (int i = 0; i < maxLen; i++) {
            if (i < bbpeTokens.length) {
                idxs[i] = vocab.get(bbpeTokens[i]).intValue();
            } else {
                idxs[i] = this.pad;
            }
        }
        return idxs;
    }

    public String[] bbpe(String unicodeToken, Map<String[], Integer> merges) {
        String[] chars = unicodeToken.split("");
        chars = mergeSP(chars);
        for (String[] pair : merges.keySet()) {
            int i = 0;
            while (i < chars.length - 1) {
                if (chars[i].equals(pair[0]) && chars[i + 1].equals(pair[1])) {
                    String pairStr = pair[0] + pair[1];
                    chars = mergeToken(chars, i, pairStr);
                } else {
                    i++;
                }
            }
        }
        return chars;
    }

    public String[] mergeSP(String[] chars) {
        List<String> mergeSp = new ArrayList<String>();
        for (int i = 0; i < chars.length; i++) {
            String tmp = "";
            if (i + this.sos_str().length() <= chars.length) {
                for (int j = 0; j < this.sos_str().length(); j++) {
                    tmp += chars[i + j];
                }
            }
            if (tmp.equals(this.sos_str())) {
                mergeSp.add(this.sos_str());
                i += this.sos_str().length() - 1;
            } else {
                mergeSp.add(chars[i]);
            }
        }
        String[] tmp1 = new String[mergeSp.size()];
        tmp1 = mergeSp.toArray(tmp1);
        mergeSp.clear();
        for (int i = 0; i < tmp1.length; i++) {
            String tmp = "";
            if (i + this.eos_str().length() <= tmp1.length) {
                for (int j = 0; j < this.eos_str().length(); j++) {
                    tmp += tmp1[i + j];
                }
            }
            if (tmp.equals(this.eos_str())) {
                mergeSp.add(this.eos_str());
                i += this.eos_str().length() - 1;
            } else {
                mergeSp.add(tmp1[i]);
            }
        }
        String[] tmp2 = new String[mergeSp.size()];
        tmp2 = mergeSp.toArray(tmp2);
        return tmp2;
    }

    private int[] encode(String txt, String charset) {
        try {
            return jb2pbToInt(txt.getBytes(charset));
        } catch (UnsupportedEncodingException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public int sos() {
        // TODO Auto-generated method stub
        return sos;
    }

    @Override
    public int eos() {
        // TODO Auto-generated method stub
        return eos;
    }

    @Override
    public int pad() {
        // TODO Auto-generated method stub
        return pad;
    }

    @Override
    public int voc_size() {
        // TODO Auto-generated method stub
        return voc_size;
    }

    @Override
    public String sos_str() {
        // TODO Auto-generated method stub
        return decoder.get(sos);
    }

    @Override
    public String eos_str() {
        // TODO Auto-generated method stub
        return decoder.get(eos);
    }

    @Override
    public String pad_str() {
        // TODO Auto-generated method stub
        return decoder.get(pad);
    }
}
