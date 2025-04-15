package com.omega.data.transformer.bertTokenizer;

import com.omega.utils.JsonUtils;
import com.omega.models.transformer.LagJsonReader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Constructs a BERT tokenizer. Based on WordPiece.
 * <p>
 * <p>
 * <p>
 * This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which
 * <p>
 * contains most of the methods. Users should refer to the superclass for more
 * <p>
 * information regarding methods.
 * <p>
 * <p>
 * <p>
 * Args:
 * <p>
 * <p>
 * <p>
 * vocab_file (:obj:`string`): File containing the vocabulary.
 * <p>
 * <p>
 * <p>
 * do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to
 * <p>
 * lowercase the input when tokenizing.
 * <p>
 * <p>
 * <p>
 * do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether
 * <p>
 * to do basic tokenization before WordPiece.
 * <p>
 * <p>
 * <p>
 * never_split (:obj:`bool`, `optional`, defaults to :obj:`True`): List of
 * <p>
 * tokens which will never be split during tokenization. Only has an effect when
 * <p>
 * :obj:`do_basic_tokenize=True`
 * <p>
 * <p>
 * <p>
 * unk_token (:obj:`string`, `optional`, defaults to "[UNK]"): The unknown
 * <p>
 * token. A token that is not in the vocabulary cannot be converted to an ID and
 * <p>
 * is set to be this token instead.
 * <p>
 * <p>
 * <p>
 * sep_token (:obj:`string`, `optional`, defaults to "[SEP]"): The separator
 * <p>
 * token, which is used when building a sequence from multiple sequences, e.g.
 * <p>
 * two sequences for sequence classification or for a text and a question for
 * <p>
 * question answering. It is also used as the last token of a sequence built
 * <p>
 * with special tokens.
 * <p>
 * <p>
 * <p>
 * pad_token (:obj:`string`, `optional`, defaults to "[PAD]"): The token used
 * <p>
 * for padding, for example when batching sequences of different lengths.
 * <p>
 * <p>
 * <p>
 * cls_token (:obj:`string`, `optional`, defaults to "[CLS]"): The classifier
 * <p>
 * token which is used when doing sequence classification (classification of the
 * <p>
 * whole sequence instead of per-token classification). It is the first token of
 * <p>
 * the sequence when built with special tokens.
 * <p>
 * <p>
 * <p>
 * mask_token (:obj:`string`, `optional`, defaults to "[MASK]"): The token used
 * <p>
 * for masking values. This is the token used when training this model with
 * <p>
 * masked language modeling. This is the token which the model will try to
 * <p>
 * predict.
 * <p>
 * <p>
 * <p>
 * tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
 * <p>
 * Whether to tokenize Chinese characters. This should likely be deactivated for
 * <p>
 * Japanese: see: https://github.com/huggingface/transformers/issues/328
 */
public class BertTokenizer implements Tokenizer {
    private final static String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+", "\\r", "\n"};
    private final static String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " ", "", ""};
    private static final int MAX_LEN = 512;
    public String unk_token = "[UNK]";
    public String sep_token = "[SEP]";
    public String pad_token = "[PAD]";
    public String cls_token = "[CLS]";
    public String mask_token = "[MASK]";
    public int sos;
    public int eos;
    public int pad;
    public int unk;
    private String vocab_file = "vocab.txt";
    private Map<String, Integer> token_id_map;
    private Map<Integer, String> id_token_map;
    private boolean do_lower_case = true;
    private boolean do_basic_tokenize = true;
    private List<String> never_split = new ArrayList<String>();
    private boolean tokenize_chinese_chars = true;
    private BasicTokenizer basic_tokenizer;
    private WordpieceTokenizer wordpiece_tokenizer;

    public BertTokenizer(String vocab_file, boolean do_lower_case, boolean do_basic_tokenize, List<String> never_split, boolean tokenize_chinese_chars) {
        this.vocab_file = vocab_file;
        this.do_lower_case = do_lower_case;
        this.do_basic_tokenize = do_basic_tokenize;
        this.never_split = never_split;
        this.tokenize_chinese_chars = tokenize_chinese_chars;
        init(vocab_file);
    }

    public BertTokenizer(String vocab_file, boolean do_lower_case, boolean tokenize_chinese_chars) {
        this.vocab_file = vocab_file;
        this.do_lower_case = do_lower_case;
        this.tokenize_chinese_chars = tokenize_chinese_chars;
        init(vocab_file);
    }

    public BertTokenizer(String vocab_file, boolean do_lower_case, boolean do_basic_tokenize, List<String> never_split, String unk_token, String sep_token, String pad_token, String cls_token, String mask_token, boolean tokenize_chinese_chars) {
        this.vocab_file = vocab_file;
        this.do_lower_case = do_lower_case;
        this.do_basic_tokenize = do_basic_tokenize;
        this.never_split = never_split;
        this.unk_token = unk_token;
        this.sep_token = sep_token;
        this.pad_token = pad_token;
        this.cls_token = cls_token;
        this.mask_token = mask_token;
        this.tokenize_chinese_chars = tokenize_chinese_chars;
        init(vocab_file);
    }

    public BertTokenizer() {
        init();
    }

    public static void main(String[] args) {
        try {
            //			String vocab_file = "H:\\transformer_dataset\\vocab.txt";
            //			boolean do_lower_case = true;
            //			boolean tokenize_chinese_chars = true;
            //
            //			BertTokenizer tokenizer = new BertTokenizer(vocab_file, do_lower_case, tokenize_chinese_chars);
            //			String datasetPath = "H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.json";
            //			String outputPath = "H:\\transformer_dataset\\wiki_idx_smallvocab.txt";
            //			String datasetPath = "H:\\transformer_dataset\\train_encyclopedia.json";
            //			String outputPath = "H:\\transformer_dataset\\medical_idx_smallvocab.txt";
            //			String datasetPath = "H:\\transformer_dataset\\563w_baidubaike.json";
            //			String outputPath = "H:\\transformer_dataset\\baike_idx_smallvocab.txt";
            //			tokenizer.encodeBaiKeDataset(datasetPath, outputPath);
            //			String[] paths = new String[] {
            //					"H:\\transformer_dataset\\wiki_idx_smallvocab.txt",
            //					"H:\\transformer_dataset\\medical_idx_smallvocab.txt",
            //					"H:\\transformer_dataset\\baike_idx_smallvocab.txt"
            //			};
            //
            //			String outpath = "H:\\transformer_dataset\\wbm_idx_smallvocab.txt";
            //			String[] paths = new String[] {
            //					"H:\\transformer_dataset\\wiki_idx_chatglm_voc.txt",
            //					"H:\\transformer_dataset\\medical_idx_chatglm_vocab.txt",
            //					"H:\\transformer_dataset\\baike_idx_chatglm_vocab.txt"
            //			};
            //
            //			String outpath = "H:\\transformer_dataset\\wbm_idx_chatglm_vocab.txt";
            //
            //			tokenizer.mergeData(paths, outpath);
            String vocab_file = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
            boolean do_lower_case = true;
            boolean tokenize_chinese_chars = true;
            BertTokenizer tokenizer = new BertTokenizer(vocab_file, do_lower_case, tokenize_chinese_chars);
            String text = "一个女孩站在山顶上，手上拿着一个苹果。";
            System.out.println(JsonUtils.toJson(tokenizer.encode(text)));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    private void init() {
        try {
            this.token_id_map = load_vocab(vocab_file);
            this.eos = this.token_id_map.get(cls_token);
            this.sos = this.token_id_map.get(sep_token);
            this.pad = this.token_id_map.get(pad_token);
            this.unk = this.token_id_map.get(unk_token);
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.id_token_map = new HashMap<Integer, String>();
        for (String key : token_id_map.keySet()) {
            this.id_token_map.put(token_id_map.get(key), key);
        }
        if (do_basic_tokenize) {
            this.basic_tokenizer = new BasicTokenizer(do_lower_case, never_split, tokenize_chinese_chars);
        }
        this.wordpiece_tokenizer = new WordpieceTokenizer(token_id_map, unk_token);
    }

    private void init(String vocab_file) {
        try {
            this.token_id_map = load_vocab_from_path(vocab_file);
            this.eos = this.token_id_map.get(sep_token);
            this.sos = this.token_id_map.get(cls_token);
            this.pad = this.token_id_map.get(pad_token);
            this.unk = this.token_id_map.get(unk_token);
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.id_token_map = new HashMap<Integer, String>();
        for (String key : token_id_map.keySet()) {
            this.id_token_map.put(token_id_map.get(key), key);
        }
        if (do_basic_tokenize) {
            this.basic_tokenizer = new BasicTokenizer(do_lower_case, never_split, tokenize_chinese_chars);
        }
        this.wordpiece_tokenizer = new WordpieceTokenizer(token_id_map, unk_token);
    }

    private Map<String, Integer> load_vocab_from_path(String vocab_file_name) throws IOException {
        FileInputStream file = new FileInputStream(vocab_file_name);
        return TokenizerUtils.generateTokenIdMap(file);
    }

    private Map<String, Integer> load_vocab(String vocab_file_name) throws IOException {
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream file = classloader.getResourceAsStream(vocab_file_name);
        return TokenizerUtils.generateTokenIdMap(file);
    }

    /**
     * Tokenizes a piece of text into its word pieces.
     * <p>
     * <p>
     * <p>
     * This uses a greedy longest-match-first algorithm to perform tokenization
     * <p>
     * using the given vocabulary.
     * <p>
     * <p>
     * <p>
     * For example: input = "unaffable" output = ["un", "##aff", "##able"]
     * <p>
     * <p>
     * <p>
     * Args: text: A single token or whitespace separated tokens. This should have
     * <p>
     * already been passed through `BasicTokenizer`.
     * <p>
     * <p>
     * <p>
     * Returns: A list of wordpiece tokens.
     */
    @Override
    public List<String> tokenize(String text) {
        List<String> split_tokens = new ArrayList<String>();
        if (do_basic_tokenize) {
            for (String token : basic_tokenizer.tokenize(text)) {
                for (String sub_token : wordpiece_tokenizer.tokenize(token)) {
                    split_tokens.add(sub_token);
                }
            }
        } else {
            split_tokens = wordpiece_tokenizer.tokenize(text);
        }
        return split_tokens;
    }

    public String convert_tokens_to_string(List<String> tokens) {
        // Converts a sequence of tokens (string) in a single string.
        return tokens.stream().map(s -> s.replace("##", "")).collect(Collectors.joining(""));
    }

    public List<Integer> convert_tokens_to_ids(List<String> tokens) {
        List<Integer> output = new ArrayList<Integer>();
        for (String s : tokens) {
            output.add(token_id_map.get(s));
        }
        return output;
    }

    public int[] tokens_to_ids(List<String> tokens) {
        int[] output = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            output[i] = token_id_map.get(tokens.get(i));
        }
        return output;
    }

    public int vocab_size() {
        return token_id_map.size();
    }

    public int[] encode(String text) {
        List<String> tokens = this.tokenize(text);
        return tokens_to_ids(tokens);
    }

    public String decode(int[] idx) {
        List<String> tokens = new ArrayList<String>();
        for (int i = 0; i < idx.length; i++) {
            tokens.add(id_token_map.get(idx[i]));
        }
        return convert_tokens_to_string(tokens);
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
                    int[] idx = encode(strTmp);
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
                    int[] idx = encode(strTmp);
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
                    int[] idx = encode(strTmp);
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
}

