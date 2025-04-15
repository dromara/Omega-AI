package com.omega.models.transformer;

import cn.hutool.core.text.csv.CsvUtil;
import cn.hutool.core.text.csv.CsvWriteConfig;
import cn.hutool.core.text.csv.CsvWriter;

import java.io.*;
import java.util.*;

public class QADataFormatter {
    public static void loadQDataCSV(String csvPath, Map<String, String> qData) {
        try (FileInputStream fin = new FileInputStream(csvPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                //	        	System.out.println(strTmp);
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    qData.put(list[0], list[1]);
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadADataCSV(String csvPath, List<Map<String, String>> aData) {
        try (FileInputStream fin = new FileInputStream(csvPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                //	        	System.out.println(strTmp);
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    Map<String, String> once = new HashMap<String, String>();
                    once.put("qid", list[1]);
                    once.put("txt", list[2]);
                    aData.add(once);
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void formatQAFormCSV(String questionPath, String answerPath, String outPath) {
        Map<String, String> question = new HashMap<String, String>();
        List<Map<String, String>> answer = new ArrayList<Map<String, String>>();
        loadQDataCSV(questionPath, question);
        loadADataCSV(answerPath, answer);
        //		System.out.println(JsonUtils.toJson(answer));
        try (FileOutputStream fos = new FileOutputStream(outPath);) {
            for (Map<String, String> once : answer) {
                String qid = once.get("qid");
                String q = question.get(qid).replaceAll(" ", "");
                String a = once.get("txt").replaceAll(" ", "");
                String text = q + " " + a;
                text += "\n";
                //					System.out.println(text);
                fos.write(text.getBytes());
            }
            fos.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void formatQA2CSV(String[] dataPaths, String outputPath) {
        try {
            List<SQLQA> qas = new ArrayList<SQLQA>();
            for (String dataPath : dataPaths) {
                List<Map<String, Object>> datas = LagJsonReader.readJsonDataSamll(dataPath);
                for (Map<String, Object> once : datas) {
                    String q = once.get("question").toString();
                    String a = once.get("query").toString();
                    SQLQA qa = new SQLQA(q, a);
                    qas.add(qa);
                }
            }
            CsvWriteConfig csvWriteConfig = new CsvWriteConfig();
            csvWriteConfig.setFieldSeparator(',');
            Map<String, String> headerAlias = new LinkedHashMap<String, String>();
            headerAlias.put("his", "历史");
            headerAlias.put("q", "问题");
            headerAlias.put("a", "回答");
            csvWriteConfig.setHeaderAlias(headerAlias);
            FileWriter fileWriter = new FileWriter(outputPath);
            CsvWriter cw = CsvUtil.getWriter(fileWriter, csvWriteConfig);
            cw.writeBeans(qas);
            cw.flush();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        //		String questionPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\question\\question.csv";
        //		String answerPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\answer\\answer.csv";
        //		String outPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\qaData.txt";
        //
        //		formatQAFormCSV(questionPath, answerPath, outPath);
        String[] dataPaths = new String[]{"H:\\sql_dataset\\full_CSpider\\CSpider\\train.json", "D:\\chroms_download\\SeSQL\\single-round-question-completed\\train.json"};
        String outputPath = "H:\\sql_dataset\\full_CSpider\\CSpider\\train_sql.csv";
        formatQA2CSV(dataPaths, outputPath);
    }

    static class SQLQA {
        private String his;
        private String q;
        private String a;

        public SQLQA(String q, String a) {
            this.q = q;
            this.a = a;
        }

        public String getQ() {
            return q;
        }

        public void setQ(String q) {
            this.q = q;
        }

        public String getA() {
            return a;
        }

        public void setA(String a) {
            this.a = a;
        }

        public String getHis() {
            return his;
        }

        public void setHis(String his) {
            this.his = his;
        }
    }
}

