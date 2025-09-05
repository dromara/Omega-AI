package com.omega.example.transformer.utils.tokenizers;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import com.omega.example.transformer.utils.bpe.BinDataType;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BatchTokenizerUtils {
	
    public static void encodeDeepSeekDatasetBPE(String dataPath, String outputPath, String vocabPath, String mergesPath, int maxLen) {
        try {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            Map<String, String> once = new HashMap<String, String>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
            int batchSize = 10000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                //		    	System.err.println(line);
                String txt = once.get("text").toString();
                if (txt.length() <= maxLen) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeExMaxLen.encode(txtList, ids, bpe, maxLen);
                        writeIn(txtList, ids, writer);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeExMaxLen.encode(txtList, ids, bpe, maxLen);
                writeIn(txtList, ids, writer);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void encodeDeepSeekFullSTFDatasetBPE(String dataPath, String outputPath, String vocabPath, String mergesPath, int maxLen) {
        try {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            Map<String, List> once = new HashMap<String, List>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
            int batchSize = 100000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                String txt = null;
                try {
                    once = JsonUtils.gson.fromJson(line, HashMap.class);
                    List txts = once.get("conversations");
                    StringBuilder sb = new StringBuilder();
                    //					sb.append("<s>system\n您好，我是人工智能机器人，请问有什么可以帮助您？</s>\n");
                    String role = "user";
                    for (int j = 0; j < txts.size(); j++) {
                        if (j % 2 == 0) {
                            role = "user";
                        } else {
                            role = "assistant";
                        }
                        Map<String, String> onceObj = (Map<String, String>) txts.get(j);
                        txt = "<s>" + role + "\n" + onceObj.get("content") + "</s>\n";
                        sb.append(txt);
                    }
                    txt = sb.toString();
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                if (txt != null) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeExMaxLen.encode(txtList, ids, bpe, maxLen);
                        writeIn(txtList, ids, writer);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeExMaxLen.encode(txtList, ids, bpe, maxLen);
                writeIn(txtList, ids, writer);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }
    
    public static void encodeDeepSeekR1STFDatasetBPE(String dataPath, String outputPath, String vocabPath, String mergesPath, int maxLen) {
        try {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            Map<String, List> once = new HashMap<String, List>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis));
            BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
            int batchSize = 10000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                String txt = null;
                try {
//                	System.out.println(line);
                	once = JsonUtils.gson.fromJson(line, HashMap.class);
                    List txts = once.get("messages");
                    StringBuilder sb = new StringBuilder();
                    //					sb.append("<s>system\n您好，我是人工智能机器人，请问有什么可以帮助您？</s>\n");
                    String role = "user";
                    for (int j = 0; j < txts.size(); j++) {
                        if (j % 2 == 0) {
                            role = "user";
                        } else {
                            role = "assistant";
                        }
                        Map<String, String> onceObj = (Map<String, String>) txts.get(j);
                        txt = "<s>" + role + "\n" + onceObj.get("content") + "</s>\n";
                        sb.append(txt);
                    }
                    txt = sb.toString();
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                if (txt != null) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeExMaxLen.encode(txtList, ids, bpe, maxLen, false);
                        writeIn(txtList, ids, writer);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeExMaxLen.encode(txtList, ids, bpe, maxLen, false);
                writeIn(txtList, ids, writer);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }
    
    public static void encodeMonkeyDatasetByBPE(String dataPath, String outputPath, String vocabPath, String mergesPath) {
        try {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            Map<String, String> once = new HashMap<String, String>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
            int batchSize = 10000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                //		    	System.err.println(line);
                String txt = once.get("text");
                txt = "<s>" + txt + "</s>";
                if (txt.length() <= 512) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeExMaxLen.encode(txtList, ids, bpe, 512);
                        //			    		write(txtList, ids, writer, bpe);
                        writeIn(txtList, ids, writer);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeExMaxLen.encode(txtList, ids, bpe, 512);
                //	    		write(txtList, ids, writer, bpe);
                writeIn(txtList, ids, writer);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void encodeMonkeyDatasetBySentencePiece(String dataPath, String outputPath, String tokenizerPath) {
        try {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            Map<String, String> once = new HashMap<String, String>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizerPath);
            int batchSize = 10000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                String txt = once.get("text");
                if (txt.length() <= 512) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeEx.encode(txtList, ids, tokenizer);
                        write(txtList, ids, writer, tokenizer);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeEx.encode(txtList, ids, tokenizer);
                write(txtList, ids, writer, tokenizer);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void encodeMonkeyDatasetBySentencePiece2Bin(String dataPath, String outputPath, String tokenizerPath, BinDataType dataType) {
        try {
            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            Map<String, String> once = new HashMap<String, String>();
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizerPath);
            int batchSize = 100000;
            List<String> txtList = new ArrayList<String>();
            String[] ids = new String[batchSize];
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                String txt = once.get("text");
                if (txt.length() <= 512) {
                    if (txt != null && !txt.equals("")) {
                        txtList.add(txt);
                    }
                    if (i > 1 && i % batchSize == 0) {
                        EncodeEx.encode(txtList, ids, tokenizer);
                        writeBin(txtList, ids, writer, tokenizer, dataType);
                        txtList.clear();
                    }
                    System.out.println(i);
                    i++;
                }
            }
            if (txtList.size() > 0) {
                EncodeEx.encode(txtList, ids, tokenizer);
                writeBin(txtList, ids, writer, tokenizer, dataType);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void pretrainTXT2Bin(String dataPath, String outputPath, String tokenizerPath, BinDataType dataType) {
        try {
            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            SentencePieceTokenizer tokenizer = new SentencePieceTokenizer(tokenizerPath);
            int batchSize = 100000;
            List<String> txtList = new ArrayList<String>();
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                txtList.add(line);
                if (i > 1 && i % batchSize == 0) {
                    writeBin(txtList, writer, tokenizer, dataType);
                    txtList.clear();
                }
                System.out.println(i);
                i++;
            }
            if (txtList.size() > 0) {
                writeBin(txtList, writer, tokenizer, dataType);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void writeIn(List<String> txtList, String[] ids, FileWriter writer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = ids[i];
            if(!txt.equals("-1")) {
            	 writer.write(txt + "\n");
            }
        }
    }

    public static void write(List<String> txtList, String[] ids, FileWriter writer, Tokenizer tokenizer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = ids[i];
            writer.write(tokenizer.sos() + " " + txt + " " + tokenizer.eos());
        }
    }

    public static void writeBin(List<String> txtList, FileOutputStream writer, Tokenizer tokenizer, BinDataType dataType) throws IOException {
        if (dataType == BinDataType.unint16) {
            writeShort(txtList, writer, tokenizer);
        } else {
            writeInt(txtList, writer, tokenizer);
        }
    }

    public static void writeBin(List<String> txtList, FileOutputStream writer, BinDataType dataType, int maxLen) throws IOException {
        if (dataType == BinDataType.unint16) {
            writeShort(txtList, writer, maxLen);
        } else {
            writeInt(txtList, writer);
        }
    }

    public static void writeLineBin(List<String> txtList, FileOutputStream writer, BinDataType dataType) throws IOException {
        if (dataType == BinDataType.unint16) {
            writeShortLine(txtList, writer);
        } else {
            writeIntLine(txtList, writer);
        }
    }

    public static void writeBin(List<String> txtList, String[] ids, FileOutputStream writer, Tokenizer tokenizer, BinDataType dataType) throws IOException {
        if (dataType == BinDataType.unint16) {
            writeShort(txtList, ids, writer, tokenizer);
        } else {
            writeInt(txtList, ids, writer, tokenizer);
        }
    }

    public static void writeShort(List<String> txtList, FileOutputStream writer) throws IOException {
        System.out.println("writing.");
        byte[] batch = new byte[2048];
        for (int i = 0; i < txtList.size(); i++) {
            String txt = txtList.get(i);
            String[] idList = txt.split(" ");
            for (int j = 0; j < idList.length; j++) {
                String str = idList[j];
                short s = Short.parseShort(str);
                byte[] bs = ModelUtils.s2b(s);
                batch[j * 2] = bs[0];
                batch[j * 2 + 1] = bs[1];
            }
            writer.write(batch);
        }
    }

    public static void writeShort(List<String> txtList, FileOutputStream writer, int maxLen) throws IOException {
        System.out.println("writing.");
        byte[] batch = new byte[maxLen * 2];
        for (int i = 0; i < txtList.size(); i++) {
            String txt = txtList.get(i);
            String[] idList = txt.split(" ");
            for (int j = 0; j < idList.length; j++) {
                String str = idList[j];
                short s = Short.parseShort(str);
                byte[] bs = ModelUtils.s2b(s);
                batch[j * 2] = bs[0];
                batch[j * 2 + 1] = bs[1];
            }
            writer.write(batch);
        }
    }

    public static void writeShortLine(List<String> txtList, FileOutputStream writer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = txtList.get(i);
            String[] idList = txt.split(" ");
            for (String str : idList) {
                short s = Short.parseShort(str);
                byte[] bs = ModelUtils.s2b(s);
                writer.write(bs);
            }
            writer.write(10);
        }
    }

    public static void writeShort(List<String> txtList, FileOutputStream writer, Tokenizer tokenizer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = tokenizer.sos() + " " + txtList.get(i) + " " + tokenizer.eos();
            String[] idList = txt.split(" ");
            for (String str : idList) {
                short s = Short.parseShort(str);
                byte[] bs = ModelUtils.s2b(s);
                writer.write(bs);
            }
        }
    }

    public static void writeShort(List<String> txtList, String[] ids, FileOutputStream writer, Tokenizer tokenizer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = tokenizer.sos() + " " + ids[i] + " " + tokenizer.eos();
            String[] idList = txt.split(" ");
            for (String str : idList) {
                short s = Short.parseShort(str);
                byte[] bs = ModelUtils.s2b(s);
                writer.write(bs);
            }
        }
    }

    public static void writeInt(List<String> txtList, FileOutputStream writer, Tokenizer tokenizer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = tokenizer.sos() + " " + txtList.get(i) + " " + tokenizer.eos();
            String[] idList = txt.split(" ");
            for (String str : idList) {
                if (!str.equals("")) {
                    int s = Integer.parseInt(str);
                    byte[] bs = ModelUtils.int2byte(s);
                    writer.write(bs);
                }
            }
        }
    }

    public static void writeInt(List<String> txtList, String[] ids, FileOutputStream writer, Tokenizer tokenizer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = tokenizer.sos() + " " + ids[i] + " " + tokenizer.eos();
            String[] idList = txt.split(" ");
            for (String str : idList) {
                if (!str.equals("")) {
                    int s = Integer.parseInt(str);
                    byte[] bs = ModelUtils.int2byte(s);
                    writer.write(bs);
                }
            }
        }
    }

    public static void writeInt(List<String> txtList, FileOutputStream writer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = txtList.get(i);
            String[] idList = txt.split(" ");
            for (String str : idList) {
                if (!str.equals("")) {
                    int s = Integer.parseInt(str);
                    byte[] bs = ModelUtils.int2byte(s);
                    writer.write(bs);
                }
            }
        }
    }

    public static void writeIntLine(List<String> txtList, FileOutputStream writer) throws IOException {
        System.out.println("writing.");
        for (int i = 0; i < txtList.size(); i++) {
            String txt = txtList.get(i);
            String[] idList = txt.split(" ");
            for (String str : idList) {
                if (!str.equals("")) {
                    int s = Integer.parseInt(str);
                    byte[] bs = ModelUtils.int2byte(s);
                    writer.write(bs);
                }
            }
            writer.write(10);
        }
    }

    public static void txt2bin(String txtPath, String binPath, int sos, int eos) {
        try {
            String line = null;
            FileInputStream fis = new FileInputStream(txtPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            File file = new File(binPath);
            if (!file.exists()) {
                try {
                    file.createNewFile();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
                int index = 0;
                while ((line = bufferedReader.readLine()) != null) {
                    String[] txts = line.split(" ");
                    int[] idx = new int[txts.length + 2];
                    idx[0] = sos;
                    idx[idx.length - 1] = eos;
                    for (int i = 1; i < idx.length - 1; i++) {
                        idx[i] = Integer.parseInt(txts[i - 1]);
                    }
                    ModelUtils.saveIntData(rFile, idx);
                    index++;
                    System.out.println(index);
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            bufferedReader.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void txt2bin(String txtPath, String binPath) {
        try {
            String line = null;
            FileInputStream fis = new FileInputStream(txtPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            File file = new File(binPath);
            if (!file.exists()) {
                try {
                    file.createNewFile();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
                int index = 0;
                while ((line = bufferedReader.readLine()) != null) {
                    String[] txts = line.split(" ");
                    int[] idx = new int[txts.length];
                    for (int i = 0; i < idx.length; i++) {
                        idx[i] = Integer.parseInt(txts[i]);
                    }
                    ModelUtils.saveIntData(rFile, idx);
                    index++;
                    System.out.println(index);
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            bufferedReader.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void txt2bin(String dataPath, String outputPath, BinDataType dataType, int maxLen) {
        try {
            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int batchSize = 100000;
            List<String> txtList = new ArrayList<String>();
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                txtList.add(line);
                if (i > 1 && i % batchSize == 0) {
                    writeBin(txtList, writer, dataType, maxLen);
                    txtList.clear();
                }
                System.out.println(i);
                i++;
            }
            if (txtList.size() > 0) {
                writeBin(txtList, writer, dataType, maxLen);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void txt2Linebin(String dataPath, String outputPath, BinDataType dataType) {
        try {
            File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);
            String line = null;
            FileInputStream fis = new FileInputStream(dataPath);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int batchSize = 100000;
            List<String> txtList = new ArrayList<String>();
            int i = 1;
            while ((line = bufferedReader.readLine()) != null) {
                txtList.add(line);
                if (i > 1 && i % batchSize == 0) {
                    writeLineBin(txtList, writer, dataType);
                    txtList.clear();
                }
                System.out.println(i);
                i++;
            }
            if (txtList.size() > 0) {
                writeLineBin(txtList, writer, dataType);
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public static void main(String[] args) {
        //		String dataPath = "H:\\transformer_dataset\\mobvoi_seq_monkey_general_open_corpus.jsonl";
        //		String outputPath = "H:\\transformer_dataset\\monkey_idx_6400_all_vocab.txt";
        //
        //		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
        //		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
        //
        //		encodeMonkeyDatasetByBPE(dataPath, outputPath, vocabPath, mergesPath);
        //		String txtPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.txt";
        //		String binPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.bin";
        //		txt2bin(txtPath, binPath, 1, 2);
        //		int time = 512;
        //
        //		int[] data = new int[time];
        //
        //		data = loadData(data, binPath);
        //
        //		BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
        //
        //		String txt = bpe.decode(data);
        //
        //		System.out.println(txt);
        //		String tokenizerPath = "H:\\transformer_dataset\\tokenizer.model";
        //		String binPath = "H:\\transformer_dataset\\monkey_idx_64793_vocab.bin";
        //
        //		encodeMonkeyDatasetBySentencePiece2Bin(dataPath, binPath, tokenizerPath, BinDataType.unint32);
        //		String txtPath = "H:\\transformer_dataset\\wbm_idx_chatglm_vocab.txt";
        //		String outputPath = "H:\\transformer_dataset\\wbm_idx_chatglm_vocab.bin";hunyuan vae train
        //
        //		pretrainTXT2Bin(txtPath, outputPath, tokenizerPath, BinDataType.unint32);
        //		String dataPath = "I:\\dataset\\sft_512.jsonl";
        //		String outputPath = "H:\\transformer_dataset\\pretrain_hq_6400.txt";
        //
        //		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
        //		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
        //
        //		encodeMonkeyDatasetByBPE(dataPath, outputPath, vocabPath, mergesPath);
        //		String txtPath = "H:\\transformer_dataset\\pretrain_hq_6400.txt";
        //		String outputPath = "H:\\transformer_dataset\\pretrain_hq_6400.bin";
        //
        //		txt2bin(txtPath, outputPath, 1, 2);
//        int maxLen = 1024;
//        String dataPath = "D:\\dataset\\llm\\qwen3_235b_thinking_2507_distill_110k.jsonl";
//        String outputPath = "D:\\dataset\\llm\\r1_mix_1024.txt";
//        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
//        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
//        encodeDeepSeekR1STFDatasetBPE(dataPath, outputPath, vocabPath, mergesPath, maxLen);
        
        int maxLen = 1024;
        String dataPath = "D:\\dataset\\llm\\qwen3_235b_thinking_2507_distill_110k.jsonl";
        String outputPath = "D:\\dataset\\llm\\r1_mix_1024.txt";
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
        encodeDeepSeekR1STFDatasetBPE(dataPath, outputPath, vocabPath, mergesPath, maxLen);
    	
//		String txtPath = "I:\\dataset\\r1_mix_1024.txt";
//		String outputPath = "H:\\transformer_dataset\\r1_mix_1024.bin";		
//		txt2bin(txtPath, outputPath, BinDataType.unint16, maxLen);
        
//		String dataPath = "H:\\transformer_dataset\\pretrain_hq.jsonl";
//		String outputPath = "H:\\transformer_dataset\\pretrain_hq_6400.txt";
        //
        //		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
        //		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
        //		encodeDeepSeekDatasetBPE(dataPath, outputPath, vocabPath, mergesPath, 512);
        //		String txtPath = "H:\\transformer_dataset\\pretrain_hq_6400.txt";
        //		String outputPath = "H:\\transformer_dataset\\pretrain_hq_6400.bin";
        //
        //		txt2bin(txtPath, outputPath, BinDataType.unint16);
    }

    public static int[] loadData(int[] data, String inputPath) {
        try (RandomAccessFile file = new RandomAccessFile(inputPath, "r")) {
            System.out.println(file.length() / 4 / 512);
            System.out.println(file.getFilePointer());
            ModelUtils.loadIntData(file, data);
            System.out.println(file.getFilePointer());
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return data;
    }
}

