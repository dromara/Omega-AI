package com.omega.data.transformer.bertTokenizer;

import java.util.ArrayList;
import java.util.List;

public class BasicTokenizer implements Tokenizer {
    private boolean do_lower_case = true;
    private List<String> never_split = new ArrayList<String>();
    private boolean tokenize_chinese_chars = true;

    public BasicTokenizer(boolean do_lower_case, List<String> never_split, boolean tokenize_chinese_chars) {
        this.do_lower_case = do_lower_case;
        if (never_split == null) {
            never_split = new ArrayList<String>();
        }
        this.tokenize_chinese_chars = tokenize_chinese_chars;
    }

    public BasicTokenizer() {
    }

    @Override
    public List<String> tokenize(String text) {
        text = TokenizerUtils.clean_text(text);
        if (tokenize_chinese_chars) {
            text = TokenizerUtils.tokenize_chinese_chars(text);
        }
        List<String> orig_tokens = TokenizerUtils.whitespace_tokenize(text);
        List<String> split_tokens = new ArrayList<String>();
        for (String token : orig_tokens) {
            if (do_lower_case && !never_split.contains(token)) {
                token = TokenizerUtils.run_strip_accents(token.toLowerCase());
                split_tokens.addAll(TokenizerUtils.run_split_on_punc(token, never_split));
            }
        }
        return TokenizerUtils.whitespace_tokenize(String.join(" ", split_tokens));
    }
}

