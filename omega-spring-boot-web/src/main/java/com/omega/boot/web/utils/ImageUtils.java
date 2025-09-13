package com.omega.boot.web.utils;

public class ImageUtils {

    public static boolean isImage(String path) {
        String[] imageSuffix = {".jpg", ".png"};
        for(String suffix : imageSuffix) {
            if(path.endsWith(suffix)) {
                return true;
            }
        }
        throw new RuntimeException("仅支持jpg、png");
    }
}
