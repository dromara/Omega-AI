package com.omega.models.transformer;

import ai.djl.sentencepiece.SpProcessor;
import ai.djl.sentencepiece.SpTokenizer;
import ai.djl.sentencepiece.SpVocabulary;
import com.omega.utils.JsonUtils;
import com.omega.models.transformer.tokenizers.Tokenizer;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SentencePieceTokenizer extends Tokenizer {
    /**
     * Unknown token (<unk>), default id 0.
     */
    public final int unk;
    /**
     * BOS (beginning of sequence) token (<s>), default id 1.
     */
    public final int bos;
    /**
     * EOS (end of sequence) token (</s>), default id 2.
     */
    public final int eos;
    public final int pad;
    /**
     * SentencePiece tokenizer.
     */
    private final SpProcessor tokenizer;
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+", "\\r", "\n"};
    private final String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " ", "", ""};
    public int voc_size;
    private SpVocabulary voc;

    public SentencePieceTokenizer(String path) throws IOException {
        System.out.println("Loading SentencePiece model from " + path);
        SpTokenizer model = new SpTokenizer(Paths.get(path));
        SpVocabulary voc = SpVocabulary.from(model);
        //        this.voc_size = (int) voc.size();
        tokenizer = model.getProcessor();
        //        for(int i = 0;i<20000;i++) {
        //        	System.out.println(i+":"+voc.getToken(i));
        //        }
        //        System.out.println(voc.getToken(52773));
        unk = (int) voc.getIndex("<unk>");
        bos = (int) voc.getIndex("<s>");
        eos = (int) voc.getIndex("</s>");
        pad = (int) voc.getIndex("<pad>");
        System.out.println("UNK ID: " + unk + " | BOS ID: " + bos + " | EOS ID: " + eos + " | PAD ID: " + pad);
    }

    public SentencePieceTokenizer(String path, int voc_size) throws IOException {
        System.out.println("Loading SentencePiece model from " + path);
        SpTokenizer model = new SpTokenizer(Paths.get(path));
        SpVocabulary voc = SpVocabulary.from(model);
        this.voc_size = voc_size;
        tokenizer = model.getProcessor();
        //        for(int i = 0;i<voc_size;i++) {
        //        	System.out.println(i+":"+voc.getToken(i));
        //        }
        unk = (int) voc.getIndex("<unk>");
        bos = (int) voc.getIndex("<s>");
        eos = (int) voc.getIndex("</s>");
        pad = (int) voc.getIndex("<pad>");
        System.out.println("UNK ID: " + unk + " | BOS ID: " + bos + " | EOS ID: " + eos + " | PAD ID: " + pad);
    }

    public SentencePieceTokenizer(String path, int voc_size, Map<Integer, String> map) throws IOException {
        System.out.println("Loading SentencePiece model from " + path);
        SpTokenizer model = new SpTokenizer(Paths.get(path));
        SpVocabulary voc = SpVocabulary.from(model);
        this.voc_size = voc_size;
        tokenizer = model.getProcessor();
        for (int i = 0; i < voc_size; i++) {
            map.put(i, voc.getToken(i));
            //        	System.out.println(i+":"+voc.getToken(i));
        }
        unk = (int) voc.getIndex("<unk>");
        bos = (int) voc.getIndex("<s>");
        eos = (int) voc.getIndex("</s>");
        pad = (int) voc.getIndex("<pad>");
        System.out.println("UNK ID: " + unk + " | BOS ID: " + bos + " | EOS ID: " + eos + " | PAD ID: " + pad);
    }

    public static void main(String[] args) {
        String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
        //    	String tokenizer_path = "H:\\transformer_dataset\\chinese_sp.model";
        try {
            SentencePieceTokenizer t = new SentencePieceTokenizer(tokenizer_path);
            //			String txt = "中国社会科学院语言研究所是中国社会科学院下设的一个汉语语言研究机构。";
            //
            //			String[] tokens = t.tokenize(txt);
            //
            //			System.out.println(JsonUtils.toJson(tokens));
            //
            //			int[] idx = t.encode(txt);
            //
            //			System.out.println(JsonUtils.toJson(idx));
            //
            //			String outText = t.decode(idx);
            //
            //			System.out.println(outText);
            //			String datasetPath = "H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.json";
            //			String outputPath = "H:\\transformer_dataset\\wiki_idx_chatglm_voc.txt";
            //
            //			t.encodeDataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\train_encyclopedia.json";
            //			String outputPath = "H:\\transformer_dataset\\medical_idx_chatglm_vocab.txt";
            //
            //			t.encodeMedicalDataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\563w_baidubaike.json";
            //			String outputPath = "H:\\transformer_dataset\\baike_idx_chatglm_vocab.txt";
            //
            //			t.encodeBaiKeDataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\alpaca_gpt4_data_zh.json";
            //			String outputPath = "H:\\transformer_dataset\\alpaca_gpt4_idx_chatglm_vocab.txt";
            //
            //			t.encodeGPT4Dataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\Belle_open_source_1M.json";
            //			String outputPath = "H:\\transformer_dataset\\Belle_open_source_1M_idx_chatglm_vocab.txt";
            //
            //			t.encodeBelleDataset(datasetPath, outputPath);
            //			int[] idx = new int[] {30910, 34234, 32718, 34283, 54532, 31679, 31930, 32114, 31884, 32654, 32136};
            //
            //			String outText = t.decode(idx);
            //
            //			System.out.println(outText);
            //			String datasetPath = "H:\\transformer_dataset\\medical\\ChatMed_Consult-v0.3.json";
            //			String outputPath = "H:\\transformer_dataset\\medical\\ChatMed_idx_chatglm_vocab.txt";
            //
            //			t.encodeChatMedDataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\medical\\DISC-Med-SFT_released.jsonl";
            //			String outputPath = "H:\\transformer_dataset\\medical\\DISC-Med-SFT_idx_chatglm_vocab.txt";
            //			t.encodeDISCDataset(datasetPath, outputPath);
            //			String datasetPath = "H:\\transformer_dataset\\medical\\HuatuoGPT_sft_data_v1.jsonl";
            //			String outputPath = "H:\\transformer_dataset\\medical\\HuatuoGPT_idx_chatglm_vocab.txt";
            //			t.encodeHuatuoDataset(datasetPath, outputPath);
            //			String[] paths = new String[] {
            //				"H:\\transformer_dataset\\alpaca_gpt4_idx_chatglm_vocab.txt",
            //				"H:\\transformer_dataset\\Belle_open_source_1M_idx_chatglm_vocab.txt"
            //			};
            //
            //			String outputSFTPath = "H:\\transformer_dataset\\sft_data_chatglm_vocab.txt";
            //
            //			t.mergeData(paths, outputSFTPath);
            //			String[] paths = new String[] {
            //				"H:\\transformer_dataset\\medical\\ChatMed_idx_chatglm_vocab.txt",
            //				"H:\\transformer_dataset\\medical\\DISC-Med-SFT_idx_chatglm_vocab.txt",
            //				"H:\\transformer_dataset\\medical\\HuatuoGPT_idx_chatglm_vocab.txt"
            //			};
            //
            //			String outputSFTPath = "H:\\transformer_dataset\\medical\\medical_sft_data_chatglm_vocab.txt";
            //
            //			t.mergeData(paths, outputSFTPath);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public int[] encodeInt(String text) {
        return encode(text, false, false);
    }

    public int[] encode(String text, boolean bos, boolean eos) {
        int[] t = tokenizer.encode(text);
        int length = t.length;
        if (bos)
            ++length;
        if (eos)
            ++length;
        int[] tokens = length > t.length ? new int[length] : t;
        if (bos) {
            tokens[0] = this.bos;
            System.arraycopy(t, 0, tokens, 1, t.length);
        } else {
            System.arraycopy(t, 0, tokens, 0, t.length);
        }
        if (eos) {
            tokens[length - 1] = this.eos;
        }
        return tokens;
    }

    public String decode(int[] tokens) {
        return tokenizer.decode(tokens);
    }

    public String[] tokenize(String text) {
        return tokenizer.tokenize(text);
    }

    public void encodeDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, String>> list = LagJsonReader.readJsonFileSamll(dataPath);
            String strTmp = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                strTmp = list.get(i).get("completion");
                for (int p = 0; p < _patterns.length; p++) {
                    strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
                }
                if (!strTmp.equals(" ") && !strTmp.equals("")) {
                    String idxStr = "";
                    int[] idx = encodeInt(strTmp);
                    for (int id : idx) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeMedicalDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, String>> list = LagJsonReader.readRowJsonFile(dataPath);
            String strTmp = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                strTmp = list.get(i).get("text");
                for (int p = 0; p < _patterns.length; p++) {
                    strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
                }
                if (!strTmp.equals(" ") && !strTmp.equals("")) {
                    String idxStr = "";
                    int[] idx = encodeInt(strTmp);
                    for (int id : idx) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeBaiKeDataset(String dataPath, String outputPath) {
        try {
            Map<String, Object> once = new HashMap<String, Object>();
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            FileReader fileReader = new FileReader(dataPath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            String strTmp = "";
            int i = 0;
            while ((line = bufferedReader.readLine()) != null) {
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
                if (!strTmp.equals(" ") && !strTmp.equals("")) {
                    strTmp.replaceAll(" ", "");
                }
                for (int p = 0; p < _patterns.length; p++) {
                    strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
                }
                if (!strTmp.equals(" ") && !strTmp.equals("")) {
                    String idxStr = "";
                    int[] idx = encodeInt(strTmp);
                    for (int id : idx) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
                i++;
            }
            bufferedReader.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public void encodeFTChatData(String dataPath, String outputPath) {
        try (FileInputStream fin = new FileInputStream(dataPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    System.out.println(JsonUtils.toJson(list));
                    int[] idx_p = encodeInt(list[0]);
                    int[] idx_a = encodeInt(list[1]);
                    int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                    System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                    idx_i[idx_p.length] = bos;
                    System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                    String idxStr = "";
                    for (int id : idx_i) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                idx++;
            }
            buffReader.close();
            writer.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        System.out.println("Data has been written to the file.");
    }

    public void encodeGPT4Dataset(String dataPath, String outputPath) {
        try {
            List<Map<String, String>> list = LagJsonReader.readJsonFileSamll(dataPath);
            String instruction = "";
            String output = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                instruction = list.get(i).get("instruction");
                output = list.get(i).get("output");
                if (!instruction.equals(" ") && !instruction.equals("")) {
                    String idxStr = "";
                    int[] idx_p = encodeInt(instruction);
                    int[] idx_a = encodeInt(output);
                    int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                    System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                    idx_i[idx_p.length] = bos;
                    System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                    for (int id : idx_i) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeBelleDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, String>> list = LagJsonReader.readRowJsonFile(dataPath);
            String instruction = "";
            String output = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                instruction = list.get(i).get("instruction");
                output = list.get(i).get("output");
                if (!instruction.equals(" ") && !instruction.equals("")) {
                    String idxStr = "";
                    int[] idx_p = encodeInt(instruction);
                    int[] idx_a = encodeInt(output);
                    int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                    System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                    idx_i[idx_p.length] = bos;
                    System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                    for (int id : idx_i) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeChatMedDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, String>> list = LagJsonReader.readRowJsonFile(dataPath);
            String instruction = "";
            String output = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                instruction = list.get(i).get("query");
                output = list.get(i).get("response");
                if (!instruction.equals(" ") && !instruction.equals("")) {
                    String idxStr = "";
                    int[] idx_p = encodeInt(instruction);
                    int[] idx_a = encodeInt(output);
                    int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                    System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                    idx_i[idx_p.length] = bos;
                    System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                    for (int id : idx_i) {
                        idxStr += id + " ";
                    }
                    writer.write(idxStr + "\n");
                }
                System.out.println(i);
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeDISCDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, Object>> list = LagJsonReader.readRowJsonFile2Obj(dataPath);
            String instruction = "";
            String output = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                List<Map<String, String>> conversation = (List<Map<String, String>>) list.get(i).get("conversation");
                if (conversation.size() >= 2) {
                    instruction = conversation.get(0).get("content");
                    output = conversation.get(1).get("content");
                    if (!instruction.equals(" ") && !instruction.equals("")) {
                        String idxStr = "";
                        int[] idx_p = encodeInt(instruction);
                        int[] idx_a = encodeInt(output);
                        int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                        System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                        idx_i[idx_p.length] = bos;
                        System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                        for (int id : idx_i) {
                            idxStr += id + " ";
                        }
                        writer.write(idxStr + "\n");
                    }
                    System.out.println(i);
                }
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void encodeHuatuoDataset(String dataPath, String outputPath) {
        try {
            List<Map<String, Object>> list = LagJsonReader.readRowJsonFile2Obj(dataPath);
            String instruction = "";
            String output = "";
            File file = new File(outputPath);
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < list.size(); i++) {
                List<String> conversation = (List<String>) list.get(i).get("data");
                if (conversation.size() >= 2) {
                    instruction = conversation.get(0).substring(2, conversation.get(0).length());
                    output = conversation.get(1).substring(2, conversation.get(1).length());
                    if (!instruction.equals(" ") && !instruction.equals("")) {
                        String idxStr = "";
                        int[] idx_p = encodeInt(instruction);
                        int[] idx_a = encodeInt(output);
                        int[] idx_i = new int[idx_p.length + idx_a.length + 1];
                        System.arraycopy(idx_p, 0, idx_i, 0, idx_p.length);
                        idx_i[idx_p.length] = bos;
                        System.arraycopy(idx_a, 0, idx_i, idx_p.length + 1, idx_a.length);
                        for (int id : idx_i) {
                            idxStr += id + " ";
                        }
                        writer.write(idxStr + "\n");
                    }
                    System.out.println(i);
                }
            }
            writer.close();
            System.out.println("Data has been written to the file.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mergeData(String[] paths, String outpath) throws IOException {
        File file = new File(outpath);
        FileWriter writer = new FileWriter(file);
        for (String path : paths) {
            try (FileReader fileReader = new FileReader(path); BufferedReader bufferedReader = new BufferedReader(fileReader);) {
                String line;
                int i = 0;
                while ((line = bufferedReader.readLine()) != null) {
                    writer.write(line + "\n");
                    System.out.println(i);
                    i++;
                }
                bufferedReader.close();
            } catch (Exception e) {
                // TODO: handle exception
            }
        }
        writer.close();
    }

    public SpVocabulary getVoc() {
        return voc;
    }

    public void setVoc(SpVocabulary voc) {
        this.voc = voc;
    }

    @Override
    public String decode(List<Integer> ids) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<Integer> encode(String txt) {
        // TODO Auto-generated method stub
        int[] idx = this.encodeInt(txt);
        List<Integer> idxs = new ArrayList<Integer>();
        for (int v : idx) {
            idxs.add(v);
        }
        return idxs;
    }

    @Override
    public int sos() {
        // TODO Auto-generated method stub
        return bos;
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
        return "<s>";
    }

    @Override
    public String eos_str() {
        // TODO Auto-generated method stub
        return "</s>";
    }

    @Override
    public String pad_str() {
        // TODO Auto-generated method stub
        return "<pad>";
    }

    @Override
    public int[] encodeInt(String txt, int maxLen) {
        // TODO Auto-generated method stub
        return null;
    }
}

