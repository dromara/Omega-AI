package com.omega.common.utils;

import com.google.gson.Gson;

public class JsonUtils {
    public static Gson gson = new Gson();

    public static String toJson(Object object) {
        String data = null;
        if (object != null) {
            data = gson.toJson(object);
        }
        return data;
    }

    public static <T> Object toObject(String jsonStr, Object obj) {
        Class<? extends Object> clazz = obj.getClass();
        if (jsonStr != null) {
            obj = gson.fromJson(jsonStr, clazz);
        }
        return obj;
    }
}

