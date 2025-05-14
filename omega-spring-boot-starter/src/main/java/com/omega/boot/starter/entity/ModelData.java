package com.omega.boot.starter.entity;

import cn.hutool.json.JSONObject;

/**
 * 模型数据实体
 *
 * @author haylee
 * @date 2025/05/14 14:33
 */
public class ModelData {

    private String path;

    private JSONObject config;

    private JSONObject tokenizerConfig;

    public ModelData() {
    }

    public ModelData(String path, JSONObject config, JSONObject tokenizerConfig) {
        this.path = path;
        this.config = config;
        this.tokenizerConfig = tokenizerConfig;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public JSONObject getConfig() {
        return config;
    }

    public void setConfig(JSONObject config) {
        this.config = config;
    }

    public JSONObject getTokenizerConfig() {
        return tokenizerConfig;
    }

    public void setTokenizerConfig(JSONObject tokenizerConfig) {
        this.tokenizerConfig = tokenizerConfig;
    }
}
