package com.omega.boot.web.utils;

import java.util.Arrays;

public class FileUtils {

    public static String getServerPath(String absolutePath) {
        int index = absolutePath.indexOf("upload");
        return absolutePath.substring(index-1);
    }
}
