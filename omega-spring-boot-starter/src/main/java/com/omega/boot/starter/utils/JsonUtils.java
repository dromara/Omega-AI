package com.omega.boot.starter.utils;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONTokener;

import java.io.FileReader;

/**
 * llama3模型初始化类
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
public class JsonUtils {

    public static JSONObject readJson(String url) throws Exception {
        // 从文件读取
        FileReader reader = new FileReader(url);
        JSONObject jsonObject = new JSONObject(new JSONTokener(reader,null));
        reader.close();
        return jsonObject;
    }


}
