package com.omega.example.transformer.utils;

import com.omega.common.utils.JsonUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class LagJsonReader {
    private final static String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+", "\\r", "\n"};
    private final static String[] _replacements = new String[]{" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " ", "", ""};

    // 读取json文件并解析为对象
    public static List<Map<String, String>> readJsonFileSamll(String path) {
        List<Map<String, String>> mapList = new ArrayList<Map<String, String>>();
        try {
            String jsonString = new String(Files.readAllBytes(Paths.get(path)));
            mapList = JsonUtils.gson.fromJson(jsonString, mapList.getClass());
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static List<Map<String, Object>> readJsonDataSamll(String path) {
        List<Map<String, Object>> mapList = new ArrayList<Map<String, Object>>();
        try {
            String jsonString = new String(Files.readAllBytes(Paths.get(path)));
            mapList = JsonUtils.gson.fromJson(jsonString, mapList.getClass());
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Map<String, Object> readJsonFileSmallWeight(String path) {
        Map<String, Object> mapList = new LinkedHashMap<String, Object>();
        try {
        	System.out.println(path+" model load start.");
            String jsonString = new String(Files.readAllBytes(Paths.get(path)));
            mapList = JsonUtils.gson.fromJson(jsonString, mapList.getClass());
            System.out.println(path+" model load finish.");
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Map<String, Object> readJsonFileBigWeight(String path) {
        Map<String, Object> mapList = new LinkedHashMap<String, Object>();
        String line = null;
        try {
            FileReader fileReader = new FileReader(path);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            StringBuilder stringBuilder = new StringBuilder();
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                stringBuilder.append(line);
            }
            bufferedReader.close();
            String json = stringBuilder.toString();
            mapList = JsonUtils.gson.fromJson(json, mapList.getClass());
            return mapList;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static List<Map<String, String>> readJsonFile(String path) {
        List<Map<String, String>> mapList = new ArrayList<Map<String, String>>();
        String line = null;
        try {
            FileReader fileReader = new FileReader(path);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            StringBuilder stringBuilder = new StringBuilder();
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                stringBuilder.append(line);
            }
            bufferedReader.close();
            String json = stringBuilder.toString();
            mapList = JsonUtils.gson.fromJson(json, mapList.getClass());
            return mapList;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static List<Map<String, String>> readRowJsonFile(String path) {
        List<Map<String, String>> mapList = new ArrayList<Map<String, String>>();
        String line = null;
        try {
            FileReader fileReader = new FileReader(path);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            StringBuilder stringBuilder = new StringBuilder();
            Map<String, String> once = new HashMap<String, String>();
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                mapList.add(once);
            }
            bufferedReader.close();
            return mapList;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static List<Map<String, Object>> readRowJsonFile2Obj(String path) {
        List<Map<String, Object>> mapList = new ArrayList<Map<String, Object>>();
        String line = null;
        try {
            FileReader fileReader = new FileReader(path);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            StringBuilder stringBuilder = new StringBuilder();
            Map<String, Object> once = new HashMap<String, Object>();
            while ((line = bufferedReader.readLine()) != null) {
                //		    	System.out.println(line);
                once = JsonUtils.gson.fromJson(line, HashMap.class);
                mapList.add(once);
            }
            bufferedReader.close();
            return mapList;
        } catch (IOException e) {
            System.out.println(line);
            e.printStackTrace();
        }
        return null;
    }

    public static void loadDataForJson(String dataPath, String txtPath) {
        List<Map<String, String>> list = LagJsonReader.readJsonFileSamll(dataPath);
        String strTmp = "";
        try {
            FileWriter fileWriter = new FileWriter(txtPath);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            for (int i = 0; i < list.size(); i++) {
                strTmp = list.get(i).get("completion");
                for (int p = 0; p < _patterns.length; p++) {
                    strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
                }
                if (!strTmp.equals(" ") && !strTmp.equals("")) {
                    bufferedWriter.write(strTmp);
                    if (i < list.size() - 1) {
                        bufferedWriter.newLine();
                    }
                }
            }
            bufferedWriter.close();
            fileWriter.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        //		List<Map<String,String>> mapList = readJsonFile("H:\\transformer_dataset\\563w_baidubaike.json\\563w_baidubaike.json");
        //		List<Map<String,String>> mapList = readJsonFileSamll("H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.json");
        //		System.out.println(JsonUtils.toJson(mapList.get(0)));
        String dataPath = "H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.json";
        String txtPath = "H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.txt";
        loadDataForJson(dataPath, txtPath);
    }
}

