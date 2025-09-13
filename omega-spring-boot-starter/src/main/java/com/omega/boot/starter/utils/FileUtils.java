package com.omega.boot.starter.utils;

import java.io.File;

public class FileUtils {

    public static String mkdir(String path) {
        File file = new File(path);
        if (!file.exists()) {
            file.mkdirs();
        }
        return path;
    }
}
