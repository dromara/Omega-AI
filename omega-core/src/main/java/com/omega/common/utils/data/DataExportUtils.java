package com.omega.common.utils.data;

import com.omega.utils.MathUtils;
import com.omega.utils.PrintUtils;

import java.io.*;

public class DataExportUtils {
    public static void exportTXT(int[][] x, String path) {
        FileOutputStream out = null;
        File file = null;
        try {
            if (path != null) {
                file = new File(path);
            }
            out = new FileOutputStream(file);
            exportTxtByOS(out, x);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void importTXT(int[][] x, String path) {
        FileInputStream fin = null;
        BufferedReader buffReader = null;
        try {
            fin = new FileInputStream(path);
            InputStreamReader reader = new InputStreamReader(fin);
            buffReader = new BufferedReader(reader);
            String strTmp = "";
            int row = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                String[] once = strTmp.split(" ");
                for (int i = 0; i < once.length; i++) {
                    x[row][i] = Integer.parseInt(once[i]);
                }
                row++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            if (fin != null) {
                try {
                    fin.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            if (buffReader != null) {
                try {
                    buffReader.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
    }

    public static boolean exportTxtByOS(OutputStream out, int[][] x) {
        boolean isSucess = false;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        try {
            osw = new OutputStreamWriter(out);
            bw = new BufferedWriter(osw);
            // 循环数据
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[i].length; j++) {
                    bw.append(x[i][j] + "").append(" ");
                }
                bw.append("\n");
            }
            isSucess = true;
        } catch (Exception e) {
            e.printStackTrace();
            isSucess = false;
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                    bw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                    osw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (out != null) {
                try {
                    out.close();
                    out = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return isSucess;
    }

    public static void main(String args[]) {
        String path = "H://index3.txt";
        int[][] indexs = MathUtils.randomInts(60000, 128);
        DataExportUtils.exportTXT(indexs, path);
        int[][] t = new int[468][128];
        DataExportUtils.importTXT(t, path);
        PrintUtils.printImage(t);
    }
}

