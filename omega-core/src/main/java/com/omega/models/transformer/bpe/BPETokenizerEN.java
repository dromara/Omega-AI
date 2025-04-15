package com.omega.models.transformer.bpe;

import com.omega.utils.JsonUtils;
import com.omega.utils.MatrixUtils;
import com.omega.models.transformer.tokenizers.Tokenizer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * bpe tokenizer
 *
 * @author Administrator
 */
public class BPETokenizerEN extends Tokenizer {
    private final static String w = "</w>";
    public Map<String, Integer> vocab;
    public Map<Integer, String> decoder = new HashMap<Integer, String>();
    public Map<String[], Integer> merges;
    public Map<Integer, String> unicodeMap;
    public int voc_size = 0;
    public int pad = 0;
    public int sos = 1;
    public int eos = 2;
    private Pattern pattern;
    private String pat = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";

    public BPETokenizerEN(String vocabPath, String mergesPath) {
        System.out.println("init bpe tokenizer.");
        this.unicodeMap = unicodeMap();
        this.vocab = readJsonFileSamll(vocabPath);
        this.voc_size = vocab.size();
        this.merges = readMerges(mergesPath);
        for (String key : vocab.keySet()) {
            decoder.put(vocab.get(key).intValue(), key);
        }
        pattern = Pattern.compile(pat);
    }

    public BPETokenizerEN(String vocabPath, String mergesPath, int sos, int eos) {
        System.out.println("init bpe tokenizer.");
        this.sos = sos;
        this.eos = eos;
        this.unicodeMap = unicodeMap();
        this.vocab = readJsonFileSamll(vocabPath);
        this.voc_size = vocab.size();
        this.merges = readMerges(mergesPath);
        for (String key : vocab.keySet()) {
            decoder.put(vocab.get(key).intValue(), key);
        }
        pattern = Pattern.compile(pat);
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

    public static String cleanUp(String text) {
        return text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re");
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
        int asciiValue = (int) ch;
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
            String vocabPath = "H:\\model\\bpe_tokenizer\\vocab.json";
            String mergesPath = "H:\\model\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
            String txt = "tiktok live stream banner, selling mens fashion clothing, casual style, whiteboard animation, brand named 大风车, clean lines, simple shapes, vibrant color palette, dynamic text, playful animation, high resolution, sharp focus, energetic atmosphere.";
            int[] ids = bpe.encodeInt(txt, 77);
            System.err.println(JsonUtils.toJson(ids));
            String decodeTxt = bpe.decode(ids);
            System.out.println(decodeTxt);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public String decode(List<Integer> ids) {
        String txt = "";
        for (int i = 0; i < ids.size(); i++) {
            String bstr = decoder.get(ids.get(i));
            txt += bstr;
        }
        return cleanUp(new String(unicode2Byte(txt, unicodeMap), StandardCharsets.UTF_8).replaceAll(w, " ").trim());
    }

    public String decode(int[] ids) {
        String txt = "";
        for (int i = 0; i < ids.length; i++) {
            String bstr = decoder.get(ids[i]);
            txt += bstr;
        }
        //		System.err.println(txt);
        return cleanUp(new String(unicode2Byte(txt, unicodeMap), StandardCharsets.UTF_8).replaceAll(w, " ").trim());
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

    public List<Integer> encode(String txt) {
        Matcher m = pattern.matcher(txt);
        List<String> list = new ArrayList<String>();
        while (m.find()) {
            list.add(m.group());
        }
        String[] bbpeTokens = bbpe(list, merges);
        List<Integer> idxs = new ArrayList<Integer>();
        for (String key : bbpeTokens) {
            idxs.add(vocab.get(key).intValue());
        }
        return idxs;
    }

    public int[] encodeInt(String txt) {
        Matcher m = pattern.matcher(txt);
        List<String> list = new ArrayList<String>();
        while (m.find()) {
            list.add(m.group());
        }
        String[] bbpeTokens = bbpe(list, merges);
        int[] idxs = new int[bbpeTokens.length];
        for (int i = 0; i < bbpeTokens.length; i++) {
            String key = bbpeTokens[i];
            idxs[i] = vocab.get(key).intValue();
        }
        return idxs;
    }

    public int[] encodeInt(String txt, int maxLen) {
        txt = txt.toLowerCase();
        Matcher m = pattern.matcher(txt);
        List<String> list = new ArrayList<String>();
        while (m.find()) {
            list.add(m.group());
        }
        String[] bbpeTokens = bbpe(list, merges);
        //		System.err.println(list.size());
        int[] idxs = new int[maxLen];
        idxs[0] = sos;
        for (int i = 1; i < maxLen; i++) {
            if (i - 1 < bbpeTokens.length) {
                //				if(vocab.get(bbpeTokens[i - 1]) == null) {
                //					System.err.println(bbpeTokens.length);
                //					System.err.println(i - 1);
                //					System.err.println(txt);
                //					System.err.println(JsonUtils.toJson(bbpeTokens));
                //					System.err.println(bbpeTokens[i - 1]);
                //					System.err.println(vocab.get(bbpeTokens[i - 1]));
                //				}
                idxs[i] = vocab.get(bbpeTokens[i - 1]).intValue();
            } else {
                idxs[i] = eos;
            }
        }
        return idxs;
    }

    public String[] bbpe(List<String> txts, Map<String[], Integer> merges) {
        List<String> list = new ArrayList<String>();
        for (String txt : txts) {
            txt = unicodeToken(encode(txt, "utf-8"), unicodeMap);
            String[] chars = txt.split("");
            chars[chars.length - 1] = chars[chars.length - 1] + w;
            for (String v : chars) {
                list.add(v);
            }
        }
        String[] chars = new String[list.size()];
        list.toArray(chars);
        //		System.err.println(JsonUtils.toJson(chars));
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

