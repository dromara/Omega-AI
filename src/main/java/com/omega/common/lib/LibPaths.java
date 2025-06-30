package com.omega.common.lib;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarInputStream;

public class LibPaths {

    private static String LIB_PATH = "H:\\omega-ai-cu\\";

    static {

        try {
            File file = new File(LIB_PATH);
            if(!file.exists()){

                // 获取 ProtectionDomain 和 CodeSource
                ProtectionDomain protectionDomain = LibPaths.class.getProtectionDomain();
                CodeSource codeSource = protectionDomain.getCodeSource();
                if (codeSource != null) {
                    String jarPath = codeSource.getLocation().getPath();
                    jarPath = jarPath.replaceFirst("^nested:", "");
                    System.out.println("JAR 文件路径: " + jarPath);
                    if(jarPath != null){
                        int index = jarPath.indexOf(".jar");
                        String destDir = jarPath;
                        if(index != -1){
                            destDir = new File(jarPath.substring(0, index+4)).getParent();
                        }

                        if(jarPath.endsWith(".jar")){
                            copyJarCuToOut(jarPath,destDir, false);
                        }else if(jarPath.endsWith(".jar!/")){
                            copyJarCuToOut(jarPath,destDir, true);
                        }

                        LIB_PATH = destDir + File.separator + "cu";
                    }else{
                        System.out.println("JAR 文件路径为空");
                    }
                } else {
                    System.out.println("无法获取 JAR 路径（可能来自系统类加载器）");
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void copyJarCuToOut(String jarPath,String destDir, boolean multiple) throws IOException {
        File file = new File(destDir + File.separator + "cu");
        if(!file.exists()){
            file.mkdirs();
            if(multiple){
                copyResourcesFromMultipleJar(jarPath, "cu", file.getPath());
            }else{
                copyResourcesFromJar(jarPath, "cu", file.getPath());
            }

        }else{
            System.out.println(file.getPath() + "已存在");
        }
    }

    /**
     * 将 JAR 中指定路径下的所有文件复制到目标目录
     *
     * @param jarFilePath   JAR 文件的路径
     * @param resourcePath  JAR 中的资源路径（例如 "resources/"）
     * @param destDir       目标输出目录
     * @throws IOException  如果复制过程中出现错误
     */
    public static void copyResourcesFromJar(String jarFilePath, String resourcePath, String destDir) throws IOException {
        try (JarFile jarFile = new JarFile(jarFilePath)) {

            Enumeration<JarEntry> entries = jarFile.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                if (entry.getName().startsWith(resourcePath) && !entry.isDirectory() && ( entry.getName().endsWith(".cu") || entry.getName().endsWith(".cu.h"))) {

                    // 构建目标路径
                    Path destPath = Paths.get(destDir + File.separator + entry.getName().substring(resourcePath.length()));

                    // 复制文件
                    try (InputStream is = jarFile.getInputStream(entry);
                         OutputStream os = Files.newOutputStream(destPath)) {
                        byte[] buffer = new byte[1024];
                        int len;
                        while ((len = is.read(buffer)) > 0) {
                            os.write(buffer, 0, len);
                        }
                    }
                    System.out.println("已复制: " + entry.getName() + " 到 " + destPath);
                }
            }
        }
    }

    /**
     * 将 JAR 中指定路径下的所有文件复制到目标目录
     *
     * @param jarFilePath   JAR 文件的路径
     * @param resourcePath  JAR 中的资源路径（例如 "resources/"）
     * @param destDir       目标输出目录
     * @throws IOException  如果复制过程中出现错误
     */
    public static void copyResourcesFromMultipleJar(String jarFilePath, String resourcePath, String destDir) throws IOException {
        // 1. 分离路径部分
        String[] parts = jarFilePath.split("/!");
        if (parts.length < 2) {
            throw new IllegalArgumentException("无效的嵌套 JAR 路径格式");
        }

        // 2. 解析外层 JAR 路径（去掉开头的 / 和 !/）
        String outerJarPath = parts[0];
        String innerJarEntry = parts[1].replace("!/", "");

        System.out.println("###################" + outerJarPath);
        // 3. 打开外层 JAR
        try (JarFile outerJar = new JarFile(outerJarPath)) {
            // 4. 获取内层 JAR 的 Entry
            JarEntry innerEntry = outerJar.getJarEntry(innerJarEntry);
            if (innerEntry == null) {
                throw new FileNotFoundException("内层 JAR 不存在: " + innerJarEntry);
            }
            System.out.println("innerJarEntry###################" + innerJarEntry);

            try (InputStream innerIs = outerJar.getInputStream(innerEntry); JarInputStream innerJar = new JarInputStream(innerIs)) {

                JarEntry entry;
                while ((entry = innerJar.getNextJarEntry()) != null) {

                    if (entry.getName().startsWith(resourcePath) && !entry.isDirectory()) {
                        // 构建目标路径
                        Path destPath = Paths.get(destDir + File.separator + entry.getName().substring(resourcePath.length()));

                        // 复制文件
                        try (OutputStream os = Files.newOutputStream(destPath)) {
                            byte[] buffer = new byte[1024];
                            int len;
                            while ((len = innerJar.read(buffer)) > 0) {
                                os.write(buffer, 0, len);
                            }
                        }
                        System.out.println("已复制: " + entry.getName() + " 到 " + destPath);
                    }
                }
            }

        }

    }

    public static String getLibPath() {
        return LIB_PATH;
    }

}