package com.omega.example.transformer.utils.bpe;

import com.omega.common.utils.JsonUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class BPETokenizer2 {
    public Map<Integer, String> vocab = new LinkedHashMap<Integer, String>();
    public LinkedHashMap<String, Integer> merge = new LinkedHashMap<String, Integer>();
    private Map<String, Integer> pairStats = new ConcurrentHashMap<String, Integer>();
    private int maxCount = 0;
    private int minCount = 0;
    private String maxPair = null;
    private String minPair = null;

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

    public static byte pb2jb(int b) {
        if (b > 127) {
            b = b - 256;
        }
        return (byte) b;
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

    public static List<String> readJsonFile(String path) {
        List<String> mapList = new ArrayList<String>();
        String line = null;
        try {
            System.out.println("start load json data.");
            FileInputStream fis = new FileInputStream(path);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            Map<String, String> once = new HashMap<String, String>();
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, once.getClass());
                //		    	System.out.println(line);
                String value = new String(once.get("text").getBytes("utf-8"), StandardCharsets.UTF_8);
                //		    	System.err.println(value);
                mapList.add(value);
            }
            bufferedReader.close();
            System.out.println("load json data finish.");
            return mapList;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static List<Integer> readJsonFileToIds(String path) {
        List<Integer> ids = new ArrayList<Integer>();
        String line = null;
        try {
            System.out.println("start load json data.");
            FileInputStream fis = new FileInputStream(path);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            Map<String, String> once = new HashMap<String, String>();
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, once.getClass());
                System.out.println(line);
                //		    	System.out.println(line);
                byte[] b = once.get("text").getBytes("utf-8");
                //		    	System.err.println(value);
                for (int j = 0; j < b.length; j++) {
                    ids.add(jb2pbToInt(b[j]));
                }
            }
            bufferedReader.close();
            System.out.println("load json data finish.");
            return ids;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    //	public void getStatsAndMax(List<Integer> ids){
    //		pairStats.clear();
    //		pairPos.clear();
    //		maxCount = 0;
    //		for(int i = 0;i<ids.size() - 1;i++) {
    //			String pairKey = ids.get(i) + ":" + ids.get(i + 1);
    //			if(pairStats.get(pairKey)!=null) {
    //				pairStats.put(pairKey, pairStats.get(pairKey) + 1);
    //				pairPos.put(pairKey, pairPos.get(pairKey)+","+i);
    //			}else {
    //				pairStats.put(pairKey, 1);
    //				pairPos.put(pairKey, i+"");
    //			}
    //			int current = pairStats.get(pairKey);
    //			if(maxCount < current) {
    //				maxCount = current;
    //				maxPair = pairKey;
    //			}
    //			System.err.println(ids.size() + ":" + i);
    //		}
    //	}
    public static List<Integer> readTxtToIds(String path) {
        List<Integer> ids = new ArrayList<Integer>();
        String line = null;
        try {
            System.out.println("start load json data.");
            FileInputStream fis = new FileInputStream(path);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                //		    	System.out.println(line);
                byte[] b = line.getBytes("utf-8");
                //		    	System.err.println(value);
                for (int j = 0; j < b.length; j++) {
                    ids.add(jb2pbToInt(b[j]));
                }
            }
            bufferedReader.close();
            System.out.println("load txt data finish.");
            return ids;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String args[]) {
        try {
            //			String trainDataPath = "H:\\transformer_dataset\\tokenizer_train.jsonl";
            String trainDataPath = "H:\\transformer_dataset\\gpt-dataset\\qaData.txt";
            BPETokenizer2 tokenizer = new BPETokenizer2();
            tokenizer.bpe(trainDataPath, 6400);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public Map<Integer, String> buildVocab() {
        for (int i = 0; i < 256; i++) {
            vocab.put(i, i + "");
        }
        //		System.out.println(JsonUtils.toJson(vocab));
        return vocab;
        //		System.out.println(JsonUtils.toJson(vocab));
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
        System.out.println("start encode to bytes.");
        List<Integer> ids = new ArrayList<Integer>();
        for (int i = 0; i < org_tokens.size(); i++) {
            byte[] b = org_tokens.get(i).getBytes(StandardCharsets.UTF_8);
            for (int j = 0; j < b.length; j++) {
                ids.add(jb2pbToInt(b[j]));
            }
            //			System.out.println((i + 1) / (count+1) * 100 + "%");
        }
        System.out.println("encode to bytes finish.");
        //		System.out.println(ids.get(100));
        //		System.out.println(ids.size());
        //		System.out.println(Collections.max(ids));
        //		System.out.println(Collections.min(ids));
        return ids;
    }

    public void getStatsAndMin(List<Integer> ids) {
        pairStats.clear();
        minCount = 99999999;
        minPair = null;
        for (int i = 0; i < ids.size() - 1; i++) {
            String pairKey = ids.get(i) + ":" + ids.get(i + 1);
            pairStats.put(pairKey, pairStats.getOrDefault(pairKey, 0) + 1);
            int current = pairStats.get(pairKey);
            if (minCount > current) {
                minCount = current;
                minPair = pairKey;
            }
        }
    }

    public List<String> loadTxt(String dataPath, int max_len) {
        List<String> org_tokens = new ArrayList<String>();
        try (FileInputStream fin = new FileInputStream(dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            //		    int dic_index = 0;
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                //		        for(int i = 0;i<_patterns.length;i++) {
                //		        	strTmp = strTmp.replaceAll(_patterns[i], _replacements[i]);
                //		        }
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

    public List<Integer> merge(List<Integer> ids, Map<String, String> pairPos, String pair, int idx) {
        String[] pos = pairPos.get(pair).split(",");
        for (String p : pos) {
            int pi = Integer.parseInt(p);
            ids.set(pi, idx);
            ids.remove(pi + 1);
        }
        return ids;
    }

    public List<Integer> merge2(List<Integer> ids, String pair, int idx) {
        List<Integer> rids = new ArrayList<Integer>();
        for (int i = 0; i < ids.size() - 1; i++) {
            String pairKey = ids.get(i) + ":" + ids.get(i + 1);
            if (pair == pairKey) {
                rids.add(i);
            }
        }
        for (int pi : rids) {
            ids.set(pi, idx);
            ids.remove(pi + 1);
        }
        return ids;
    }

    public Map<Integer, String> bpe(String dataPath, int num_merges) {
        buildVocab();
        //		List<Integer> ids = readJsonFileToIds(dataPath);
        List<Integer> ids = readTxtToIds(dataPath);
        //		List<Integer> ids = getIdx(txts);
        int idx = 256;
        int it = num_merges - 256;
        for (int i = 0; i < it; i++) {
            long start = System.nanoTime();
            System.out.println("current:" + ((i + 1.0f) / it * 100) + "%");
            if (ids.size() <= 1) {
                break;
            }
            long start1 = System.nanoTime();
            //			getStatsAndMax(ids);
            maxPair = MaxPairEx.getMaxKey(ids, pairStats);
            System.out.println(maxPair);
            System.out.println("1:" + (System.nanoTime() - start1) / 1e6 + "ms.");
            idx = 256 + i;
            long start3 = System.nanoTime();
            ids = MergeEx.merge(ids, maxPair, idx);
            //			ids = merge2(ids, maxPair, idx);
            System.out.println("2:" + (System.nanoTime() - start3) / 1e6 + "ms.");
            //			System.out.println(ids.size());
            int p1 = Integer.parseInt(maxPair.split(":")[0]);
            int p2 = Integer.parseInt(maxPair.split(":")[1]);
            merge.put(maxPair, idx);
            vocab.put(idx, vocab.get(p1) + ":" + vocab.get(p2));
            System.out.println("costTime:{" + (System.nanoTime() - start) / 1e6 + "ms.},vocab[" + vocab.size() + "],merge[" + merge.size() + "]");
        }
        //		System.out.println(JsonUtils.toJson(ids));
        return vocab;
    }

    public List<Integer> encode(String text) {
        List<Integer> ids = getIdx(text);
        while (ids.size() >= 2) {
            getStatsAndMin(ids);
            if (minPair == null) {
                break;
            }
            Integer idx = merge.get(minPair);
            ids = MergeEx.merge(ids, maxPair, idx);
            //            ids = merge(ids, pairPos, minPair, idx);
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
}

