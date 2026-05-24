package com.omega.boot.web.utils;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class FileUtils {

    public static String getServerPath(String absolutePath) {
        int index = absolutePath.indexOf("upload");
        return absolutePath.substring(index-1);
    }

    public static String getFileName(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return "";
        }
        Path path = Paths.get(filePath);
        return path.getFileName().toString();
    }
}
