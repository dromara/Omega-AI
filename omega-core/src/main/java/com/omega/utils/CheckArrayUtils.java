package com.omega.utils;

public class CheckArrayUtils {
    public static boolean allCheck(float[][][][] x, float[][][][] y) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int m = 0; m < x[i][j].length; m++) {
                    for (int n = 0; n < x[i][j][m].length; n++) {
                        if (x[i][j][m][n] != y[i][j][m][n]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    public static float check(float[][][][] x, float[][][][] y) {
        //		int errorCount = 0;
        float error = 0.0f;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int m = 0; m < x[i][j].length; m++) {
                    for (int n = 0; n < x[i][j][m].length; n++) {
                        //						if(Math.abs((x[i][j][m][n] - y[i][j][m][n])) > 0) {
                        ////							System.err.println(x[i][j][m][n]+ ":" + y[i][j][m][n]);
                        //							errorCount++;
                        //						}
                        error += Math.abs((x[i][j][m][n] - y[i][j][m][n]));
                    }
                }
            }
        }
        //		System.err.println("errorCount:"+errorCount);
        return error;
    }

    public static float check(float[][] x, float[][] y) {
        float error = 0.0f;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                error += Math.abs((x[i][j] - y[i][j]));
            }
        }
        return error;
    }

    public static float check(float[] x, float[] y) {
        float error = 0.0f;
        for (int i = 0; i < x.length; i++) {
            error += Math.abs((x[i] - y[i]));
        }
        return error;
    }

    public static float oneCheck(float[] x, float[] y) {
        float error = 0.0f;
        int index = 0;
        for (int i = 0; i < x.length; i++) {
            float val = Math.abs((x[i] - y[i]));
            if (error <= val) {
                error = val;
                index = i;
            }
        }
        System.out.println(x[index] + ":" + y[index]);
        return error;
    }
}

