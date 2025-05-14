package com.omega.boot.starter.utils;

import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.type.classreading.CachingMetadataReaderFactory;
import org.springframework.core.type.classreading.MetadataReader;
import org.springframework.core.type.classreading.MetadataReaderFactory;
import org.springframework.util.ClassUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 类扫描工具类
 *
 * @author haylee
 * @date 2025/05/14 14:33
 */
public class PackageScannerUtils {

    public static List<Class<?>> scanClasses(String basePackage) throws IOException, ClassNotFoundException {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        MetadataReaderFactory metadataReaderFactory = new CachingMetadataReaderFactory(resolver);

        String packageSearchPath = "classpath*:" +
                ClassUtils.convertClassNameToResourcePath(basePackage) + "/**/*.class";

        List<Class<?>> classes = new ArrayList<>();
        for (Resource resource : resolver.getResources(packageSearchPath)) {
            MetadataReader metadataReader = metadataReaderFactory.getMetadataReader(resource);
            String className = metadataReader.getClassMetadata().getClassName();
            classes.add(Class.forName(className));
        }
        return classes;
    }

    public static Map<String,Class<?>> scanClassesMap(String packageName) throws IOException, ClassNotFoundException {
        Map<String,Class<?>> classesMap = new HashMap<>();
        List<Class<?>> classes = scanClasses(packageName);
        for (Class<?> clazz : classes) {
            if(clazz.getSimpleName() != null && !clazz.getSimpleName().equals("")){
                classesMap.put(clazz.getSimpleName().toLowerCase(),clazz);
            }
        }
        return classesMap;
    }

    public static void main(String[] args) {
        try {
            String packageName = "com.omega.example.transformer.utils";
            List<Class<?>> classes = scanClasses(packageName);
            for (Class<?> clazz : classes) {
                System.out.println(clazz.getSimpleName());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
