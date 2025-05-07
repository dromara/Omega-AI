package com.omega.example.transformer.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class BPETokenizer {
    private static final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+"};
    private static final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
    public LinkedHashMap<String, Integer> merge = new LinkedHashMap<String, Integer>();
    public Map<Integer, String> vocab = new LinkedHashMap<Integer, String>();
    public Map<Integer, String> decode_vocab = new LinkedHashMap<Integer, String>();
    public Map<String, Integer> specials = new LinkedHashMap<String, Integer>();

    public BPETokenizer() {
    }

    public BPETokenizer(String vocabPath, String decoder) {
        loadVocab(vocabPath, decoder);
        loadMerges();
        System.out.println(JsonUtils.toJson(merge));
        System.out.println(merge.size());
    }

    public static int pickTopN(float[] x, int n) {
        float[] sort = Arrays.copyOf(x, x.length);
        Arrays.sort(sort);
        float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
        float v = topN[RandomUtils.getRandomNumber(topN)];
        for (int i = 0; i < x.length; i++) {
            if (v == x[i]) {
                return i;
            }
        }
        return 0;
    }

    private static byte[] listTobyte1(List<Byte> list) {
        if (list == null || list.size() < 0)
            return null;
        byte[] bytes = new byte[list.size()];
        int i = 0;
        Iterator<Byte> iterator = list.iterator();
        while (iterator.hasNext()) {
            bytes[i] = iterator.next();
            i++;
        }
        return bytes;
    }

    public static void main(String[] args) {
        //		String dataPath = "H:\\transformer_dataset\\gpt\\50w.txt";
        //
        //		Map<Integer,String> vocab = bpe(dataPath, 5000);
        //
        //		System.out.println(JsonUtils.toJson(vocab));
        //		System.out.println(vocab.size());
        //		String txt = "唉想起下午的测试，萧炎轻叹了一口气，懒懒的抽回手掌，双手枕着脑袋，眼神有些恍惚十五年了呢低低的自喃声，忽然毫无边际的从少年嘴中轻吐了出来。在萧炎的心中，有一个仅有他自己知道的秘密：他并不是这个世界的人，或者说，萧炎的灵魂，并不属于这个世界，他来自一个名叫地球的蔚蓝星球，至于为什么会来到这里，这种离奇经过，他也无法解释，不过在生活了一段时间之后，他还是后知后觉的明白了过来：他穿越了！随着年龄的增长，对这块大陆，萧炎也是有了些模糊的了解";
        //		String txt = "你好吗，你好呀，我的朋友";
        //		BPETokenizer tokenizer = new BPETokenizer();
        //
        ////		Map<Integer,String> vocab = tokenizer.bpeTxt(txt, 300);
        //		Map<Integer,String> vocab = tokenizer.bpe(dataPath, 6000);
        //		System.out.println(JsonUtils.toJson(vocab));
        //		System.out.println(vocab.size());
        //
        //		List<Integer> codes = tokenizer.encode(txt);
        //
        //		System.out.println(JsonUtils.toJson(codes));
        //
        //		System.out.println(tokenizer.decode(codes));
        //		String vocabPath = "H:\\transformer_dataset\\gpt\\50w_vocab.json";
        //
        //		String decoderPath = "H:\\transformer_dataset\\gpt\\50w_decode_vocab.json";
        //
        //		BPETokenizer t = new BPETokenizer(vocabPath, decoderPath);
        //
        //		List<Integer> codes = t.encode(txt);
        //		System.out.println("txt encode:"+JsonUtils.toJson(codes));
        //		System.out.println(t.decode(codes));
        //
        //		Map<Integer,String> decodeVocab = t.decodeVocab();
        //		System.out.println(JsonUtils.toJson(decodeVocab));
        //		byte[] bt = new byte[] {233-255,145-255};
        //		try {
        //			System.out.println(new String(bt, "utf-8"));
        //		} catch (UnsupportedEncodingException e) {
        //			// TODO Auto-generated catch block
        //			e.printStackTrace();
        //		}
        //		List<Integer> sort = Collections.synchronizedList(new ArrayList<Integer>(10));
        //		IntStream.range(0, 10).parallel().forEach(action->{
        //			sort.add(action,action);
        //		});
        //		System.out.println(JsonUtils.toJson(sort));
        String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
        try {
            Map<Integer, String> vocab = new HashMap<Integer, String>();
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizer_path, 64789, vocab);
            System.out.println(JsonUtils.toJson(vocab));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void loadMerges() {
        for (Integer key : vocab.keySet()) {
            if (key >= 256) {
                merge.put(vocab.get(key), key);
            }
        }
    }

    public List<Integer> getIdx(String txt) {
        List<Integer> ids = new ArrayList<Integer>();
        byte[] b = txt.getBytes(StandardCharsets.UTF_8);
        //		System.out.println(b.length);
        //		System.out.println(JsonUtils.toJson(b));
        for (int j = 0; j < b.length; j++) {
            ids.add(jb2pbToInt(b[j]));
        }
        return ids;
    }

    public List<Integer> getIdx(List<String> org_tokens) {
        //		System.out.println(org_tokens.size());
        List<Integer> ids = new ArrayList<Integer>();
        for (int i = 0; i < org_tokens.size(); i++) {
            byte[] b = org_tokens.get(i).getBytes(StandardCharsets.UTF_8);
            for (int j = 0; j < b.length; j++) {
                ids.add(jb2pbToInt(b[j]));
            }
        }
        //		System.out.println(ids.get(100));
        //		System.out.println(ids.size());
        //		System.out.println(Collections.max(ids));
        //		System.out.println(Collections.min(ids));
        return ids;
    }

    public List<String> loadTxt(String dataPath, int max_len) {
        List<String> org_tokens = new ArrayList<String>();
        try (FileInputStream fin = new FileInputStream(dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            //				int dic_index = 0;
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                for (int i = 0; i < _patterns.length; i++) {
                    strTmp = strTmp.replaceAll(_patterns[i], _replacements[i]);
                }
                strTmp = strTmp.toLowerCase();
                strTmp = strTmp.substring(0, strTmp.length() - 1);
                if (!strTmp.equals(" ") && !strTmp.equals("") && strTmp.length() <= max_len - 2) {
                    org_tokens.add(strTmp);
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return org_tokens;
    }

    public Map<Integer, String> buildVocab() {
        for (int i = 0; i < 256; i++) {
            vocab.put(i, i + "");
        }
        //		System.out.println(JsonUtils.toJson(vocab));
        return vocab;
        //		System.out.println(JsonUtils.toJson(vocab));
    }

    public String jb2pb(int b) {
        if (b < 0) {
            b = b + 256;
        }
        return b + "";
    }

    public int jb2pbToInt(int b) {
        if (b < 0) {
            b = b + 256;
        }
        return b;
    }

    public byte pb2jb(int b) {
        if (b > 127) {
            b = b - 256;
        }
        return (byte) b;
    }

    //	public String getMin(LinkedHashMap<String, Integer> map) {
    //		String minKey = null;
    //		int min = 999999999;
    //		for(String key:merge.keySet()) {
    //			if(map.containsKey(key)) {
    //				int current = map.get(key);
    //				if(current < min) {
    //					min = current;
    //					minKey = key;
    //					System.out.println("minKey:"+minKey+"["+current+"]");
    //				}
    //			}
    //		}
    //		return minKey;
    //	}
    //	public String getMax(Map<String, Integer> map) {
    //		Optional<Map.Entry<String, Integer>> m0 = map.entrySet().stream().max(Map.Entry.comparingByValue());
    //		return m0.get().getKey();
    //	}
    //	public String getMin(Map<String, Integer> map) {
    //		Optional<Map.Entry<String, Integer>> m0 = map.entrySet().stream().min(Map.Entry.comparingByValue());
    //		return m0.get().getKey();
    //	}
    public Map<String, Integer> getStats(List<Integer> ids) {
        Map<String, Integer> counts = Collections.synchronizedMap(new LinkedHashMap<String, Integer>());
        counts.clear();
        //		IntStream.range(0, ids.size() - 1).parallel().forEach(i->{
        //			String pairKey = ids.get(i) + ":" + ids.get(i + 1);
        //			if(counts.get(pairKey)!=null) {
        //				counts.put(pairKey, counts.get(pairKey) + 1);
        //			}else {
        //				counts.put(pairKey, 1);
        //			}
        //		});
        for (int i = 0; i < ids.size() - 1; i++) {
            String pairKey = ids.get(i) + ":" + ids.get(i + 1);
            if (counts.get(pairKey) != null) {
                counts.put(pairKey, counts.get(pairKey) + 1);
            } else {
                counts.put(pairKey, 1);
            }
        }
        return counts;
    }

    public String getMax(Map<String, Integer> map) {
        String maxKey = null;
        int max = 0;
        for (String key : map.keySet()) {
            int current = map.get(key);
            if (current > max) {
                max = current;
                maxKey = key;
            }
        }
        return maxKey;
    }

    public String getMin(Map<String, Integer> map) {
        String minKey = null;
        int min = 999999999;
        for (String key : map.keySet()) {
            if (merge.containsKey(key)) {
                int current = merge.get(key);
                if (current < min) {
                    min = current;
                    minKey = key;
                    //					System.out.println("minKey:"+minKey+"["+current+"]");
                }
            }
        }
        return minKey;
    }

    public List<Integer> merge(List<Integer> ids, String pair, int idx) {
        List<Integer> new_ids = new ArrayList<Integer>();
        int i = 0;
        int p1 = Integer.parseInt(pair.split(":")[0]);
        int p2 = Integer.parseInt(pair.split(":")[1]);
        //		Stream<Integer> steam = ids.parallelStream();
        while (i < ids.size()) {
            if (ids.get(i) == p1 && i < ids.size() - 1 && ids.get(i + 1) == p2) {
                new_ids.add(idx);
                i += 2;
            } else {
                new_ids.add(ids.get(i));
                i += 1;
            }
        }
        return new_ids;
    }

    public Map<Integer, String> bpe(String dataPath, int num_merges) {
        buildVocab();
        List<String> txts = loadTxt(dataPath, 1024);
        List<Integer> ids = getIdx(txts);
        int idx = 256;
        for (int i = 0; i < num_merges; i++) {
            long start = System.nanoTime();
            System.out.println("current:" + ((i + 1.0f) / num_merges * 100) + "%");
            if (ids.size() <= 1) {
                break;
            }
            long start1 = System.nanoTime();
            Map<String, Integer> stats = getStats(ids);
            System.out.println("1:" + (System.nanoTime() - start1) / 1e6 + "ms.");
            long start2 = System.nanoTime();
            String pair = getMax(stats);
            System.out.println("2:" + (System.nanoTime() - start2) / 1e6 + "ms.");
            idx = 256 + i;
            long start3 = System.nanoTime();
            ids = merge(ids, pair, idx);
            System.out.println("3:" + (System.nanoTime() - start3) / 1e6 + "ms.");
            int p1 = Integer.parseInt(pair.split(":")[0]);
            int p2 = Integer.parseInt(pair.split(":")[1]);
            merge.put(pair, idx);
            vocab.put(idx, vocab.get(p1) + ":" + vocab.get(p2));
            System.out.println("costTime:{" + (System.nanoTime() - start) / 1e6 + "ms.},vocab[" + vocab.size() + "],merge[" + merge.size() + "]");
        }
        //		System.out.println(JsonUtils.toJson(ids));
        return vocab;
    }

    public Map<Integer, String> bpeTxt(String txt, int num_merges) {
        buildVocab();
        List<Integer> ids = getIdx(txt);
        //		System.out.println("ids:"+JsonUtils.toJson(ids));
        //		System.out.println("vocab:"+JsonUtils.toJson(vocab));
        int idx = 256;
        for (int i = 0; i < num_merges; i++) {
            if (ids.size() <= 1) {
                break;
            }
            Map<String, Integer> stats = getStats(ids);
            String pair = getMax(stats);
            idx = 256 + i;
            ids = merge(ids, pair, idx);
            int p1 = Integer.parseInt(pair.split(":")[0]);
            int p2 = Integer.parseInt(pair.split(":")[1]);
            merge.put(pair, idx);
            vocab.put(idx, vocab.get(p1) + ":" + vocab.get(p2));
        }
        //		System.out.println("idsed:"+JsonUtils.toJson(ids));
        //		System.out.println(ids.size());
        //		System.out.println(merge);
        return vocab;
    }

    public Map<Integer, String> decodeVocab() {
        Map<Integer, String> newVocab = new HashMap<Integer, String>();
        System.out.println(JsonUtils.toJson(vocab));
        for (Integer key : vocab.keySet()) {
            String bstr = vocab.get(key);
            List<Byte> bytes = new ArrayList<Byte>();
            if (key == 5642) {
                System.out.println(bstr);
            }
            for (String bstro : bstr.split(":")) {
                bytes.add(pb2jb(Integer.parseInt(bstro)));
            }
            newVocab.put(key, new String(listTobyte1(bytes), StandardCharsets.UTF_8));
        }
        return newVocab;
    }

    public List<Integer> encode(String text) {
        List<Integer> ids = getIdx(text);
        while (ids.size() >= 2) {
            Map<String, Integer> stats = getStats(ids);
            String pair = getMin(stats);
            //            System.out.println(pair);
            if (pair == null) {
                break;
            }
            Integer idx = merge.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    public String decode(List<Integer> ids) {
        List<Byte> bytes = new ArrayList<Byte>();
        for (int i = 0; i < ids.size(); i++) {
            String bstr = vocab.get(ids.get(i));
            for (String bstro : bstr.split(":")) {
                bytes.add(pb2jb(Integer.parseInt(bstro)));
            }
        }
        return new String(listTobyte1(bytes), StandardCharsets.UTF_8);
    }

    public String toText(List<Integer> ids) {
        String result = "";
        for (int i = 0; i < ids.size(); i++) {
            //			System.out.println(ids.get(i));
            String bstr = decode_vocab.get(ids.get(i));
            result += bstr;
        }
        return result;
    }

    public String decode(Tensor output) {
        List<Byte> bytes = new ArrayList<Byte>();
        for (int i = 0; i < output.number; i++) {
            int charIndex = pickTopN(output.getByNumber(i), 1);
            String bstr = vocab.get(charIndex);
            for (String bstro : bstr.split(":")) {
                bytes.add(pb2jb(Integer.parseInt(bstro)));
            }
        }
        return new String(listTobyte1(bytes), StandardCharsets.UTF_8);
    }

    public String toText(Tensor output) {
        String result = "";
        for (int i = 0; i < output.number; i++) {
            int charIndex = pickTopN(output.getByNumber(i), 1);
            String bstr = decode_vocab.get(charIndex);
            result += bstr;
        }
        return result;
    }

    public Map<Integer, String> loadVocab(String filepath, String decoder) {
        try {
            File file = new File(filepath);
            File decoder_file = new File(decoder);
            if (file.exists() && decoder_file.exists()) {
                Map<String, String> org_vocab = new LinkedHashMap<String, String>();
                Map<String, String> de_vocab = new LinkedHashMap<String, String>();
                try (FileInputStream fos = new FileInputStream(file); Reader reader = new InputStreamReader(fos, "utf-8");) {
                    int ch = 0;
                    StringBuffer sb = new StringBuffer();
                    while ((ch = reader.read()) != -1) {
                        sb.append((char) ch);
                    }
                    String json = sb.toString();
                    org_vocab = JsonUtils.gson.fromJson(json, org_vocab.getClass());
                    for (String key : org_vocab.keySet()) {
                        Integer keyInt = Integer.parseInt(key);
                        vocab.put(keyInt, org_vocab.get(key));
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                try (FileInputStream fos = new FileInputStream(decoder_file); Reader reader = new InputStreamReader(fos, "utf-8");) {
                    int ch = 0;
                    StringBuffer sb = new StringBuffer();
                    while ((ch = reader.read()) != -1) {
                        sb.append((char) ch);
                    }
                    String json = sb.toString();
                    de_vocab = JsonUtils.gson.fromJson(json, org_vocab.getClass());
                    for (String key : org_vocab.keySet()) {
                        Integer keyInt = Integer.parseInt(key);
                        decode_vocab.put(keyInt, de_vocab.get(key));
                    }
                    System.out.println(JsonUtils.toJson(decode_vocab));
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                return vocab;
            } else {
                throw new RuntimeException("the config file is not exists.");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public void addSpecial(String val) {
        specials.put(val, vocab.size());
        byte[] b = val.getBytes(StandardCharsets.UTF_8);
        String bs = "";
        for (int i = 0; i < b.length; i++) {
            byte o = b[i];
            if (i == 0) {
                bs = o + "";
            } else {
                bs += ":" + o;
            }
        }
        vocab.put(vocab.size(), bs);
        decode_vocab.put(decode_vocab.size(), val);
    }
}

