package com.omega.example.transformer.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import com.google.gson.stream.JsonReader;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.tensor.Tensor;

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
                System.err.println(name);
                Object value = readValue(reader);
                mapList.put(name, value);
                System.out.println(name+"==>finish.");
            }
            reader.endObject();
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    public static void readJsonFileBigWeightIterator(String path, Map<String, Tensor> weights, List<String> igones) {

        try (FileInputStream fis = new FileInputStream(path);
            JsonReader reader = new JsonReader(new InputStreamReader(fis, "UTF-8"))) {
            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                System.err.println("name:"+name);
                if(!igones.contains(name)) {
                	 Object value = readValue(reader);
                     Tensor weight = weights.get(name);
                     if(weight != null) {
                     	loadData(weight, value);
                     }else {
                    	 weight = loadData(weight, value, 1);
                    	 weights.put(name, weight);
                     }
                     System.out.println(name+"==>finish.");
                }else {
                	 Object value = readValue(reader);
                     System.out.println(name+"==>igone.");
                }
               
            }
            reader.endObject();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public static void readJsonFileBigWeightIterator_5dims(String path, Map<String, Tensor> weights, List<String> igones) {

        try (FileInputStream fis = new FileInputStream(path);
            JsonReader reader = new JsonReader(new InputStreamReader(fis, "UTF-8"))) {
            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                System.err.println("name:"+name);
                if(!igones.contains(name)) {
                	 Object value = readValue(reader);
                     Tensor weight = weights.get(name);
//                     System.err.println("name:"+name+"=="+weight);
                     if(weight != null) {
                     	loadData_5dims(weight, value);
                     }else {
                    	 weight = loadData(weight, value, 1);
                    	 weights.put(name, weight);
                     }
                     System.out.println(name+"==>finish.");
                }else {
                	 Object value = readValue(reader);
                     System.out.println(name+"==>igone.");
                }
               
            }
            reader.endObject();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public static int getDim(Tensor x) {
        int dim = 0;
        if (x.number > 1) {
            dim++;
        }
        if (x.channel > 1) {
            dim++;
        }
        if (x.height > 1) {
            dim++;
        }
        if (x.width > 1) {
            dim++;
        }
        return dim;
    }
    
    public static void loadData(Tensor x, Object meta) {
        if (meta != null) {
            int dim = getDim(x);
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            }else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                if(!x.checkShape(N, C, H, W)) {
                	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C + "," + H + "," + W +"]。");
                }
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.data[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
        }
    }
    
    
    public static boolean try5dims(Object meta) {
    	try {
    		 List<List<List<List<List<Double>>>>> dataA = (List<List<List<List<List<Double>>>>>) meta;
             int N = dataA.size();
             int C = dataA.get(0).size();
             int F = dataA.get(0).get(0).size();
             int H = dataA.get(0).get(0).get(0).size();
             int W = dataA.get(0).get(0).get(0).get(0).size();
    		 return true;
		} catch (Exception e) {
			// TODO: handle exception
			return false;
		}
    }
    
    public static void loadData_5dims(Tensor x, Object meta) {
        if (meta != null) {
            int dim = getDim(x);
            if(try5dims(meta)) {
            	dim = 5;
            }
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            }else {
                List<List<List<List<List<Double>>>>> dataA = (List<List<List<List<List<Double>>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int F = dataA.get(0).get(0).size();
                int H = dataA.get(0).get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).get(0).size();
                if(!x.checkShape(N, C * F, H, W)) {
                	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C * F + "," + H + "," + W +"]。");
                }
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                    	for (int f = 0; f < F; f++) {
	                        for (int h = 0; h < H; h++) {
	                            for (int w = 0; w < W; w++) {
	                                x.data[n * x.getOnceSize() + c * F * H * W + f * H * W + h * W + w] = dataA.get(n).get(c).get(f).get(h).get(w).floatValue();
	                            }
	                        }
                    	}
                    }
                }
            }
            x.hostToDevice();
        }
    }
    
    public static Tensor loadData(Tensor x, Object meta, int dim) {
        if (meta != null) {
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                x = new Tensor(1, 1, 1, dataA.size(), true);
                int N = dataA.size();
                if(x != null) {
                	if(!x.checkShape(1, 1, 1, N)) {
                    	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ 1 + "," + 1 + "," + 1 + "," + N +"]。");
                    }
                }
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                x = new Tensor(dataA.size(), 1, 1, dataA.get(0).size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
                //				float[][] data = (float[][]) meta;
                //				x.data = MatrixUtils.transform(data);
            } else if (dim == 3) {
                List<List<List<Double>>> dataA = (List<List<List<Double>>>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    for (int h = 0; h < dataA.get(n).size(); h++) {
                        for (int w = 0; w < dataA.get(n).get(h).size(); w++) {
                            x.data[n * x.getOnceSize() + h * x.width + w] = dataA.get(n).get(h).get(w).floatValue();
                        }
                    }
                }
            } else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                x = new Tensor(N, C, H, W, true);
                if(x != null) {
                	if(!x.checkShape(N, C, H, W)) {
                    	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C + "," + H + "," + W +"]。");
                    }
                }
                for (int n = 0; n < dataA.size(); n++) {
                    for (int c = 0; c < dataA.get(n).size(); c++) {
                        for (int h = 0; h < dataA.get(n).get(c).size(); h++) {
                            for (int w = 0; w < dataA.get(n).get(c).get(h).size(); w++) {
                                x.data[n * x.getOnceSize() + c * x.height * x.width + h * x.width + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            return x;
        }
        return null;
    }
    
    public static Map<String, Object> readJsonBig(String path) throws IOException{
    	try (Stream<Map.Entry<String, Object>> stream = JsonFileIterator.streamOf(path)) {
    	    stream
    	        .filter(entry -> entry.getKey().matches(""))  // 过滤需要的 key
    	        .parallel()  // ⚠️ 谨慎使用并行：需确保 process 线程安全
    	        .forEach(entry -> {
    	            String userId = entry.getKey();
    	            System.err.println(userId);
//    	            Map<String, Object> userData = (Map<String, Object>) entry.getValue();
//    	            saveToDatabase(userId, userData);
    	        });
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

