package com.omega.boot.starter.utils;

import com.omega.boot.starter.service.Yolov3Service;

import java.io.File;
import java.security.CodeSource;
import java.security.ProtectionDomain;

public class JarUrlUtils {

    public static String getJarPath() {
        ProtectionDomain protectionDomain = JarUrlUtils.class.getProtectionDomain();
        CodeSource codeSource = protectionDomain.getCodeSource();
        String destDir = System.getProperty("user.dir");
        if(codeSource != null){
            String jarPath = codeSource.getLocation().getPath();
            jarPath = jarPath.replaceFirst("^nested:", "");
            if(jarPath != null){
                int index = jarPath.indexOf(".jar");
                if(index != -1){
                    destDir = new File(jarPath.substring(0, index+4)).getParent();
                }
            }

        }
        return destDir;
    }
}
