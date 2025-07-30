package com.omega.example.transformer.utils;

import com.google.gson.stream.JsonReader;
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

    public static Map<String, Object> readJsonFileBigWeightIterator(String path) {
        Map<String, Object> mapList = new LinkedHashMap<>();
        try (FileInputStream fis = new FileInputStream(path);
             JsonReader reader = new JsonReader(new InputStreamReader(fis, "UTF-8"))) {

            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                Object value = readValue(reader);
                mapList.put(name, value);
            }
            reader.endObject();
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static Object readValue(JsonReader reader) throws IOException {
        switch (reader.peek()) {
            case STRING:
                return reader.nextString();
            case NUMBER:
                // 根据实际情况处理整数或浮点数
                String numStr = reader.nextString();
                if (numStr.contains(".")) {
                    return Double.parseDouble(numStr);
                } else {
                    return Long.parseLong(numStr);
                }
            case BOOLEAN:
                return reader.nextBoolean();
            case NULL:
                reader.nextNull();
                return null;
            case BEGIN_ARRAY:
                return readArray(reader);
            case BEGIN_OBJECT:
                return readObject(reader);
            default:
                throw new IllegalStateException("Unexpected token: " + reader.peek());
        }
    }

    private static List<Object> readArray(JsonReader reader) throws IOException {
        List<Object> list = new ArrayList<>();
        reader.beginArray();
        while (reader.hasNext()) {
            list.add(readValue(reader));
        }
        reader.endArray();
        return list;
    }

    private static Map<String, Object> readObject(JsonReader reader) throws IOException {
        Map<String, Object> map = new LinkedHashMap<>();
        reader.beginObject();
        while (reader.hasNext()) {
            String name = reader.nextName();
            map.put(name, readValue(reader));
        }
        reader.endObject();
        return map;
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

