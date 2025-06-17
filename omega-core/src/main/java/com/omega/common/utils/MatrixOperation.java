package com.omega.common.utils;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.pooling.PoolingType;

import java.util.Vector;

/**
 * @author lijiaming
 * @ClassName: MatrixOperation
 * @date 2020年7月24日
 * @Description: TODO(用一句话描述该文件做什么)
 */
public class MatrixOperation {
    private static final int threadNum = 8;

    /**
     * @param x
     * @return
     * @throws @Title: exp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] exp(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.exp(x[i]);
        }
        return temp;
    }

    public static float[] log(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (x[i] == 0) {
                temp[i] = (float) Math.log(1e-47);
            } else {
                temp[i] = (float) Math.log(x[i]);
            }
        }
        return temp;
    }

    public static float[] sin(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.sin(x[i]);
        }
        return temp;
    }

    public static float[] cos(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.cos(x[i]);
        }
        return temp;
    }

    public static float[] tan(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.tan(x[i]);
        }
        return temp;
    }

    public static float[] atan(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.atan(x[i]);
        }
        return temp;
    }

    public static float[] tan_back(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) (1.0f / Math.pow(Math.cos(x[i]), 2));
        }
        return temp;
    }

    public static float[] atan_back(float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = 1.0f / (1.0f + x[i] * x[i]);
        }
        return temp;
    }

    public static boolean isNaN(float[] x) {
        for (float v : x) {
            if (Float.isNaN(v)) {
                return true;
            }
        }
        return false;
    }

    public static boolean isInfinite(float[] x) {
        for (float v : x) {
            if (Float.isInfinite(v)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param x
     * @return
     * @throws @Title: exp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] pow(float[] x, float e) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = (float) Math.pow(x[i], e);
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: exp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] pow(float[][] x, float e) {
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = (float) Math.pow(x[i][j], e);
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] add(float[] x, float b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] + b;
        }
        return temp;
    }

    public static float[] addToY(float[] x, float b, int n, int c, int h, int w, int[] position) {
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        switch (dims) {
            case 0:
                return addByNumberToY(x, b, n, c, h, w, start, count);
            case 1:
                return addByChannelToY(x, b, n, c, h, w, start, count);
        }
        return null;
    }

    public static float[] addByNumberToY(float[] x, float b, int n, int c, int h, int w, int start, int count) {
        int size = c * h * w;
        float[] y = MatrixUtils.clone(x);
        for (int i = 0; i < count; i++) {
            int tn = i / size + start;
            int tc = (i / h / w) % c;
            int th = i / w;
            int tw = i % h;
            int index = tn * size + tc * h * w + th * w + tw;
            y[index] = x[index] + b;
        }
        return y;
    }

    public static float[] addByChannelToY(float[] x, float b, int n, int c, int h, int w, int start, int count) {
        int size = c * h * w;
        int bc = count / n / h / w;
        float[] y = MatrixUtils.clone(x);
        for (int i = 0; i < count; i++) {
            int tn = i / size;
            int tc = (i / h / w) % bc + start;
            int th = i / w;
            int tw = i % h;
            int index = tn * size + tc * h * w + th * w + tw;
            y[index] = x[index] + b;
        }
        return y;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] add(float[][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] + b;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] add(float[][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int n = 0; n < x[i][j].length; n++) {
                    temp[i][j][n] = x[i][j][n] + b;
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] add(float[][][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        temp[n][c][h][w] = x[n][c][h][w] + b;
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] add(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] + b[i];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void plus(float[] x, float[] b) {
        for (int i = 0; i < x.length; i++) {
            x[i] += b[i];
        }
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void plus(float[] x, float[] b, int axis) {
        for (int i = 0; i < x.length; i++) {
            int xi = i / axis;
            x[xi] += b[i];
        }
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void add(float[] x, float[] b, int n, int c, int h, int w, int[] position) {
        int dims = position[0];
        int start = position[1];
        switch (dims) {
            case 0:
                addByNumber(x, b, n, c, h, w, start);
                break;
            case 1:
                addByChannel(x, b, n, c, h, w, start);
                break;
        }
    }

    public static void addByNumber(float[] x, float[] b, int n, int c, int h, int w, int start) {
        int size = c * h * w;
        for (int i = 0; i < b.length; i++) {
            int tn = i / size + start;
            int tc = (i / h / w) % c;
            int th = (i / w) % h;
            int tw = i % h;
            int index = tn * size + tc * h * w + th * w + tw;
            x[index] = x[index] + b[i];
        }
    }

    public static void addByChannel(float[] x, float[] b, int n, int c, int h, int w, int start) {
        int bc = b.length / n / h / w;
        int size = bc * h * w;
        for (int i = 0; i < b.length; i++) {
            int tn = i / size;
            int tc = (i / h / w) % bc + start;
            int th = (i / w) % h;
            int tw = i % h;
            int index = tn * c * h * w + tc * h * w + th * w + tw;
            x[index] = x[index] + b[i];
        }
    }

    public static float[] addToY(float[] x, float[] b, int n, int c, int h, int w, int[] position) {
        int dims = position[0];
        int start = position[1];
        switch (dims) {
            case 0:
                return addByNumberToY(x, b, n, c, h, w, start);
            case 1:
                return addByChannelToY(x, b, n, c, h, w, start);
        }
        return null;
    }

    public static float[] addByNumberToY(float[] x, float[] b, int n, int c, int h, int w, int start) {
        int size = c * h * w;
        float[] y = MatrixUtils.clone(x);
        for (int i = 0; i < b.length; i++) {
            int tn = i / size + start;
            int tc = (i / h / w) % c;
            int th = i / w;
            int tw = i % h;
            int index = tn * size + tc * h * w + th * w + tw;
            y[index] = x[index] + b[i];
        }
        return y;
    }

    public static float[] addByChannelToY(float[] x, float[] b, int n, int c, int h, int w, int start) {
        int size = c * h * w;
        int bc = b.length / n / h / w;
        float[] y = MatrixUtils.clone(x);
        for (int i = 0; i < b.length; i++) {
            int tn = i / size;
            int tc = (i / h / w) % bc + start;
            int th = i / w;
            int tw = i % h;
            int index = tn * size + tc * h * w + th * w + tw;
            y[index] = x[index] + b[i];
        }
        return y;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] add(float[][][][] x, float[][][][] b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        temp[n][c][h][w] = x[n][c][h][w] + b[n][c][h][w];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] add(float[][] x, float[][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] + b[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] add(float[][][] x, float[] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i][j] = x[c][i][j] + b[c];
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] add(float[][][][] x, float[] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                temp[index][c][i][j] = x[index][c][i][j] + b[c];
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * @param x
     * @param b
     * @param type
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] addByBN(float[][][][] x, float[] b, int type) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                if (type == 0) {
                                    temp[index][c][i][j] = x[index][c][i][j] + b[j];
                                } else {
                                    temp[index][c][i][j] = x[index][c][i][j] + b[c];
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: add
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] add(float[][][] x, float[][][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i][j] = x[c][i][j] + b[c][i][j];
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] subtraction(float[] x, float b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] - b;
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] subtraction(float b, float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = b - x[i];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] subtractionForOne(float[][][] x, float b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i][0][0] - b;
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] subtraction(float[][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] - b;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtraction(float[][][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        temp[n][c][h][w] = x[n][c][h][w] - b;
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] subtraction(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] - b[i];
        }
        return temp;
    }

    public static float[] subtraction(float[] x, float[] b, int axis) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            int bi = i / axis;
            temp[i] = x[i] - b[bi];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] subtractionP(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] - b[i];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] subtractionP(float[][] x, float[][] b) {
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                temp[m][i] = x[m][i] - b[m][i];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtractionP(float[][][][] x, float[][][][] b) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                for (int j = 0; j < x[m][i].length; j++) {
                    for (int n = 0; n < x[m][i][j].length; n++) {
                        temp[m][i][j][n] = x[m][i][j][n] - b[m][i][j][n];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtractionP(float[][][][] x, float[][][] b) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                for (int j = 0; j < x[m][i].length; j++) {
                    for (int n = 0; n < x[m][i][j].length; n++) {
                        temp[m][i][j][n] = x[m][i][j][n] - b[i][j][n];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @param type 0:fully,1:conv
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtraction(float[][][][] x, float[] b, int type) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                if (type == 0) {
                                    temp[index][c][i][j] = x[index][c][i][j] - b[j];
                                } else {
                                    temp[index][c][i][j] = x[index][c][i][j] - b[c];
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtractionForConv(float[][][][] x, float[] b) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                for (int j = 0; j < x[m][i].length; j++) {
                    for (int n = 0; n < x[m][i][j].length; n++) {
                        temp[m][i][j][n] = x[m][i][j][n] - b[i];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] subtraction(float[][] x, float[][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] - b[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] subtraction(float[][][] x, float[][][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    temp[k][i][j] = x[k][i][j] - b[k][i][j];
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] subtraction(float[][][][] x, float[][][][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int k = 0; k < x[c].length; k++) {
                for (int i = 0; i < x[c][k].length; i++) {
                    for (int j = 0; j < x[c][k][i].length; j++) {
                        temp[c][k][i][j] = x[c][k][i][j] - b[c][k][i][j];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] multiplication(float[] x, float b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] * b;
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void multiplication_self(float[] x, float b) {
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] * b;
        }
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] multiplication(float[][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] * b;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] multiplication(float[][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i][j] = x[c][i][j] * b;
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] multiplication(float[][][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int k = 0; k < x[c].length; k++) {
                for (int i = 0; i < x[c][k].length; i++) {
                    for (int j = 0; j < x[c][k][i].length; j++) {
                        temp[c][k][i][j] = x[c][k][i][j] * b;
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] multiplication(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] * b[i];
        }
        return temp;
    }

    public static float[] bool(float[] x, float[] b, float val) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (b[i] == 1) {
                temp[i] = val;
            } else {
                temp[i] = x[i];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @param type 0:fully,1:conv
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] multiplicationByBN(float[][][][] x, float[] gama, int type) {
        if (x == null || gama == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != gama.length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                if (type == 0) {
                                    temp[index][c][i][j] = x[index][c][i][j] * gama[j];
                                } else {
                                    temp[index][c][i][j] = x[index][c][i][j] * gama[c];
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] multiplication(float[][] x, float[][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] * b[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] multiplicationForMatrix(float[][] x, float[][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][] temp = MatrixUtils.zero(x.length, b[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < x[0].length; k++) {
                    temp[i][j] += x[i][k] * b[k][j];
                }
            }
        }
        return temp;
    }

    public static float[][] transpose(float[][] x) {
        int rows = x.length;
        int cols = x[0].length;
        float[][] y = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y[j][i] = x[i][j];
            }
        }
        return y;
    }
    /**
     *
     * @Title: multiplication
     *
     *
     *
     * @param x
     *
     * @param b
     *
     * @return
     *
     *
     *
     * @Description:
     *
     * TODO(这里用一句话描述这个方法的作用)
     *
     *
     *
     * @throws
     *
     */
    //	public static float[][] multiplicationByEjml(float[][] x,float[][] z) {
    //		int m = x.length;
    //		int n = z.length;
    //		int k = z[0].length;
    //
    //		float[][] temp = MatrixUtils.zero(m,k);
    //		SimpleMatrix a = new SimpleMatrix(x);
    //		SimpleMatrix b = new SimpleMatrix(z);
    ////		long start = System.nanoTime();
    ////		System.out.println("========start["+m+","+n+","+k+"]");
    //		SimpleMatrix c = a.mult(b);
    ////		System.out.println("========end:"+(System.nanoTime() - start) / 1e6);
    //		for(int i = 0;i<temp.length;i++) {
    //			for(int j = 0;j<temp[i].length;j++) {
    //				temp[i][j] = (float)c.get(i, j);
    //			}
    //		}
    //		return temp;
    //	}

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] multiplicationByCuda(float[][] x, float[][] z) {
        int m = x.length;
        int n = z.length;
        int k = z[0].length;
        float[] c = MatrixUtils.zero(m * k);
        float[] a = MatrixUtils.transform(x);
        float[] b = MatrixUtils.transform(z);
        GPUOP.getInstance().multiplyFloat(m, n, k, a, b, c);
        return MatrixUtils.transform(c, m, k);
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] multiplication(float[][][] x, float[][][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length || x[0][0].length != b[0][0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i][j] = x[c][i][j] * b[c][i][j];
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: multiplication
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] multiplication(float[][][][] x, float[][][][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length || x[0][0].length != b[0][0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    for (int l = 0; l < x[c][i][j].length; l++) {
                        temp[c][i][j][l] = x[c][i][j][l] * b[c][i][j][l];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] division(float[] x, float b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] / b;
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] division(float b, float[] x) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = b / x[i];
        }
        return temp;
    }

    public static float[] division(float[] x, float[] b, int axis) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            int bi = i / axis;
            temp[i] = x[i] / b[bi];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void division_self(float[] x, float b) {
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] / b;
        }
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] division(float[][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j] / b;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] division(float[][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length);
        for (int n = 0; n < x.length; n++) {
            for (int m = 0; m < x[n].length; m++) {
                for (int i = 0; i < x[n][m].length; i++) {
                    temp[n][m][i] = x[n][m][i] / b;
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] division(float[][][][] x, float b) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int n = 0; n < x.length; n++) {
            for (int m = 0; m < x[n].length; m++) {
                for (int i = 0; i < x[n][m].length; i++) {
                    for (int j = 0; j < x[n][m][i].length; j++) {
                        temp[n][m][i][j] = x[n][m][i][j] / b;
                    }
                }
            }
        }
        return temp;
    }

    public static void divisionSelf(float[][][][] x, float b) {
        for (int n = 0; n < x.length; n++) {
            for (int m = 0; m < x[n].length; m++) {
                for (int i = 0; i < x[n][m].length; i++) {
                    for (int j = 0; j < x[n][m][i].length; j++) {
                        x[n][m][i][j] /= b;
                    }
                }
            }
        }
    }

    /**
     * @param x
     * @param b
     * @param type 0:fully,1:conv
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] division(float[][][][] x, float b[], int type) {
        if (x == null) {
            new RuntimeException("matrix is null.");
        }
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                if (type == 0) {
                                    temp[index][c][i][j] = x[index][c][i][j] / b[j];
                                } else {
                                    temp[index][c][i][j] = x[index][c][i][j] / b[c];
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] division(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] / b[i];
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: division
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] division(float[][] x, float[][] b) {
        if (x == null || b == null) {
            new RuntimeException("matrix is null.");
        }
        if (x.length != b.length || x[0].length != b[0].length) {
            new RuntimeException("x size must equals b.");
        }
        float[][] temp = MatrixUtils.zero(x.length, x[0].length);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                temp[i][j] = x[i][j] / b[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] division(float[][][][] x, float[][][] b) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                for (int j = 0; j < x[m][i].length; j++) {
                    for (int n = 0; n < x[m][i][j].length; n++) {
                        temp[m][i][j][n] = x[m][i][j][n] / b[i][j][n];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: subtraction
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] division(float[][][][] x, float[][][][] b) {
        float[][][][] temp = MatrixUtils.zero(x.length, x[0].length, x[0][0].length, x[0][0][0].length);
        for (int m = 0; m < x.length; m++) {
            for (int i = 0; i < x[m].length; i++) {
                for (int j = 0; j < x[m][i].length; j++) {
                    for (int n = 0; n < x[m][i][j].length; n++) {
                        temp[m][i][j][n] = x[m][i][j][n] / b[m][i][j][n];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float dot(float[] x, float[] y) {
        float temp = 0.0f;
        for (int i = 0; i < x.length; i++) {
            temp += x[i] * y[i];
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float sum(float[] x) {
        float temp = 0.0f;
        for (float o : x) {
            temp += o;
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] sum(float[] x, int n, int c, int h, int w, int axis) {
        int count = 1;
        switch (axis) {
            case 0:
                count = 1;
                break;
            case 1:
                count = n;
                break;
        }
        float[] temp = new float[count];
        switch (axis) {
            case 0:
                for (float o : x) {
                    temp[0] += o;
                }
                break;
            case 1:
                for (int i = 0; i < x.length; i++) {
                    int b = i / c / h / w;
                    temp[b] += x[i];
                }
                break;
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] max(float[] x, int n, int c, int h, int w, int axis) {
        int count = 1;
        switch (axis) {
            case 0:
                count = 1;
                break;
            case 1:
                count = n;
                break;
        }
        float[] temp = new float[count];
        switch (axis) {
            case 0:
                temp[0] += max(x);
                break;
            case 1:
                for (int i = 0; i < n; i++) {
                    float max = -3.402823466e+38F;
                    for (int j = 0; j < c * h * w; j++) {
                        if (max <= x[i * c * h * w + j]) {
                            max = x[i * c * h * w + j];
                        }
                    }
                    temp[i] += max;
                }
                break;
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] max_backward(float[] d, float[] x, int n, int c, int h, int w, int axis) {
        float[] temp = new float[x.length];
        float max = -3.402823466e+38F;
        int max_idx = -1;
        switch (axis) {
            case 0:
                max = -3.402823466e+38F;
                for (int j = 0; j < n * c * h * w; j++) {
                    if (max <= x[j]) {
                        max = x[j];
                        max_idx = j;
                    }
                }
                temp[max_idx] = d[0];
                break;
            case 1:
                for (int i = 0; i < n; i++) {
                    max = -3.402823466e+38F;
                    for (int j = 0; j < c * h * w; j++) {
                        if (max <= x[i * c * h * w + j]) {
                            max = x[i * c * h * w + j];
                            max_idx = i * c * h * w + j;
                        }
                    }
                    temp[max_idx] = d[i];
                }
                break;
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float sum(float[][] x) {
        float temp = 0.0f;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp += x[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float sum(float[][][][] x) {
        float temp = 0.0f;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int n = 0; n < x[i][j].length; n++) {
                    for (int m = 0; m < x[i][j][n].length; m++) {
                        temp += x[i][j][n][m];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] sumByBn(float[][][][] x) {
        float[][][] temp = new float[x[0].length][x[0][0].length][x[0][0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int n = 0; n < x[i][j].length; n++) {
                    for (int m = 0; m < x[i][j][n].length; m++) {
                        temp[j][n][m] += x[i][j][n][m];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] sumBias(float[][][] x) {
        float[] temp = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int h = 0; h < x[i].length; h++) {
                for (int w = 0; w < x[i][h].length; w++) {
                    temp[i] += x[i][h][w];
                }
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws @Title: count
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] sumBias(float[][][][] x) {
        float[] temp = new float[x[0].length];
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        temp[c] += x[n][c][h][w];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * broadcast
     *
     * @param a
     * @param b
     * @param axis
     */
    public static void broadcast(float[] a, float[] b, int N, int C, int H, int W, int axis) {
        switch (axis) {
            case 0:
                for (int i = 0; i < b.length; i++) {
                    b[i] = a[0];
                }
                break;
            case 1:
                for (int i = 0; i < b.length; i++) {
                    int n = i / C / H / W;
                    b[i] = a[n];
                }
                break;
        }
    }

    /**
     * broadcast
     *
     * @param a
     * @param b
     * @param axis
     */
    public static void broadcast_plus(float[] a, float[] b, int N, int C, int H, int W, int axis) {
        switch (axis) {
            case 0:
                for (int i = 0; i < b.length; i++) {
                    b[i] += a[0];
                }
                break;
            case 1:
                for (int i = 0; i < b.length; i++) {
                    int n = i / C / H / W;
                    b[i] += a[n];
                }
                break;
        }
    }

    /**
     * isZero
     *
     * @param x
     * @return
     */
    public static boolean isZero(float[][][] x) {
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    if (x[k][i][j] != 0) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * isZero
     *
     * @param x
     * @return
     */
    public static boolean isZero(float[] x) {
        for (int k = 0; k < x.length; k++) {
            if (x[k] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return
     * @throws @Title: max
     * @Description:
     */
    public static float max(float[] x) {
        float max = -3.402823466e+38F;
        if (x.length > 0) {
            //			max = x[0];
            for (int i = 0; i < x.length; i++) {
                if (max <= x[i]) {
                    max = x[i];
                }
            }
        }
        return max;
    }

    /**
     * @return
     * @throws @Title: max
     * @Description:
     */
    public static float max(float[][][] x) {
        float max = 0.0f;
        if (x.length > 0) {
            max = x[0][0][0];
            for (int c = 0; c < x.length; c++) {
                for (int h = 0; h < x[c].length; h++) {
                    for (int w = 0; w < x[c][h].length; w++) {
                        if (max <= x[c][h][w]) {
                            max = x[c][h][w];
                        }
                    }
                }
            }
        }
        return max;
    }

    /**
     * @return
     * @throws @Title: max
     * @Description:
     */
    public static float max(float[][][][] x) {
        float max = 0.0f;
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        if (max <= x[n][c][h][w]) {
                            max = x[n][c][h][w];
                        }
                    }
                }
            }
        }
        return max;
    }

    /**
     * @return
     * @throws @Title: max
     * @Description:
     */
    public static float min(float[][][][] x) {
        float min = 0.0f;
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        if (min >= x[n][c][h][w]) {
                            min = x[n][c][h][w];
                        }
                    }
                }
            }
        }
        return min;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: clamp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] clamp(float[] x, float min, float max) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            float val = x[i];
            if (val < min) {
                temp[i] = min;
            } else if (val > max) {
                temp[i] = max;
            } else {
                temp[i] = val;
            }
        }
        return temp;
    }

    public static float[] clampSelf(float[] x, float min, float max) {
        for (int i = 0; i < x.length; i++) {
            float val = x[i];
            if (val < min) {
                x[i] = min;
            } else if (val > max) {
                x[i] = max;
            }
        }
        return x;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: maximum
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] maximum(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (x[i] >= b[i]) {
                temp[i] = x[i];
            } else {
                temp[i] = b[i];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: maximum
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] minimum(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (x[i] < b[i]) {
                temp[i] = x[i];
            } else {
                temp[i] = b[i];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: maximum
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] maximum_back(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (x[i] >= b[i]) {
                temp[i] = 1;
            } else {
                temp[i] = 0;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: maximum
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] minimum_back(float[] x, float[] b) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            if (x[i] < b[i]) {
                temp[i] = 1;
            } else {
                temp[i] = 0;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: clamp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] clamp_back(float[] x, float min, float max) {
        float[] temp = MatrixUtils.zero(x.length);
        for (int i = 0; i < x.length; i++) {
            float val = x[i];
            if (val < min || val > max) {
                temp[i] = 0;
            } else {
                temp[i] = 1;
            }
        }
        return temp;
    }

    /**
     * @param x
     * @param b
     * @return
     * @throws @Title: clamp
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] mean(float[] x, int number, int channel, int height, int width, int dim) {
        int scal = number;
        if (dim == 1) {
            scal = channel;
        }
        float[] temp = MatrixOperation.sum(x, number, channel, height, width, dim);
        MatrixOperation.division_self(temp, scal);
        return temp;
    }

    /**
     * @return
     * @throws @Title: maxIndex
     * @Description:
     */
    public static int maxIndex(float[] x) {
        int index = 0;
        if (x.length > 0) {
            float max = x[0];
            for (int i = 0; i < x.length; i++) {
                if (max <= x[i]) {
                    max = x[i];
                    index = i;
                }
            }
        }
        return index;
    }

    /**
     * @return
     * @throws @Title: maxIndex
     * @Description:
     */
    public static int maxIndex(float[][][] x) {
        int index = 0;
        if (x.length > 0) {
            float max = x[0][0][0];
            for (int i = 0; i < x.length; i++) {
                if (max <= x[i][0][0]) {
                    max = x[i][0][0];
                    index = i;
                }
            }
        }
        return index;
    }

    /**
     * mean
     *
     * @param x
     * @return
     */
    public static float[][][] mean(float[][][][] x) {
        float[][][] mean = new float[x[0].length][x[0][0].length][x[0][0][0].length];
        for (int m = 0; m < x.length; m++) {
            for (int c = 0; c < x[m].length; c++) {
                for (int h = 0; h < x[m][c].length; h++) {
                    for (int w = 0; w < x[m][c][h].length; w++) {
                        mean[c][h][w] += x[m][c][h][w] / x.length;
                    }
                }
            }
        }
        return mean;
    }

    /**
     * mean
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static float[] mean(float[][][][] x, int type) {
        int count = 0;
        float scale = 1.0f / x.length;
        if (type == 0) {
            count = x[0][0][0].length;
        } else {
            count = x[0].length;
            scale = 1.0f / (x.length * x[0][0].length * x[0][0][0].length);
        }
        float[] mean = new float[count];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int m = 0; m < x.length; m++) {
            final int index = m;
            final float s = scale;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    synchronized (mean) {
                        for (int c = 0; c < x[index].length; c++) {
                            for (int h = 0; h < x[index][c].length; h++) {
                                for (int w = 0; w < x[index][c][h].length; w++) {
                                    if (type == 0) {
                                        mean[w] += x[index][c][h][w] * s;
                                    } else {
                                        mean[c] += x[index][c][h][w] * s;
                                    }
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return mean;
    }

    public static float norm(float[] x) {
        x = MatrixOperation.pow(x, 2.0f);
        float sum = MatrixOperation.sum(x);
        return (float) Math.sqrt(sum);
    }

    /**
     * mean
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static float[] meanO(float[][][][] x, int type) {
        int count = 0;
        if (type == 0) {
            count = x[0][0][0].length;
        } else {
            count = x[0].length;
        }
        float[] mean = new float[count];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int m = 0; m < x.length; m++) {
            for (int c = 0; c < x[m].length; c++) {
                for (int h = 0; h < x[m][c].length; h++) {
                    for (int w = 0; w < x[m][c][h].length; w++) {
                        if (type == 0) {
                            mean[w] += x[m][c][h][w] / x.length;
                        } else {
                            mean[c] += x[m][c][h][w] / x.length / x[0][0].length / x[0][0][0].length;
                        }
                    }
                }
            }
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return mean;
    }

    /**
     * mean
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void mean(float[][][][] x, float[] mean, int type) {
        float scale = 1.0f / x.length;
        if (type != 0) {
            scale = 1.0f / (x.length * x[0][0].length * x[0][0][0].length);
        }
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int m = 0; m < x.length; m++) {
            final int index = m;
            final float s = scale;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    synchronized (mean) {
                        for (int c = 0; c < x[index].length; c++) {
                            for (int h = 0; h < x[index][c].length; h++) {
                                for (int w = 0; w < x[index][c][h].length; w++) {
                                    if (type == 0) {
                                        mean[w] += x[index][c][h][w] * s;
                                    } else {
                                        mean[c] += x[index][c][h][w] * s;
                                    }
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * mean
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void meanV2(float[][][][] x, float[] mean, int type) {
        float scale = 1.0f / x.length;
        if (type != 0) {
            scale = 1.0f / (x.length * x[0][0].length * x[0][0][0].length);
        }
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        if (type == 0) {
            for (int w = 0; w < x[0][0][0].length; w++) {
                final int index = w;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int c = 0; c < x[0].length; c++) {
                            for (int h = 0; h < x[0][c].length; h++) {
                                for (int n = 0; n < x.length; n++) {
                                    val += x[n][c][h][index] * s;
                                }
                            }
                        }
                        mean[index] = val;
                        return null;
                    }
                });
            }
        } else {
            for (int c = 0; c < x[0].length; c++) {
                final int index = c;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int n = 0; n < x.length; n++) {
                            for (int h = 0; h < x[n][index].length; h++) {
                                for (int w = 0; w < x[n][index][h].length; w++) {
                                    val += x[n][index][h][w] * s;
                                }
                            }
                        }
                        mean[index] = val;
                        return null;
                    }
                });
            }
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * mean
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void meanV2(float[][][][] x, float[][][][] y, float[] mean, int type) {
        float scale = 1.0f / x.length;
        if (type != 0) {
            scale = 1.0f / (x.length * x[0][0].length * x[0][0][0].length);
        }
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        if (type == 0) {
            for (int w = 0; w < x[0][0][0].length; w++) {
                final int index = w;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int c = 0; c < x[0].length; c++) {
                            for (int h = 0; h < x[0][c].length; h++) {
                                for (int n = 0; n < x.length; n++) {
                                    val += x[n][c][h][index] * y[n][c][h][index] * s;
                                }
                            }
                        }
                        mean[index] = val;
                        return null;
                    }
                });
            }
        } else {
            for (int c = 0; c < x[0].length; c++) {
                final int index = c;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int n = 0; n < x.length; n++) {
                            for (int h = 0; h < x[n][index].length; h++) {
                                for (int w = 0; w < x[n][index][h].length; w++) {
                                    val += x[n][index][h][w] * y[n][index][h][w] * s;
                                }
                            }
                        }
                        mean[index] = val;
                        return null;
                    }
                });
            }
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * standard deviation
     *
     * @param x
     * @return
     */
    public static float[][][] std(float[][][][] x) {
        float[][][] std = new float[x[0].length][x[0][0].length][x[0][0][0].length];
        std = MatrixOperation.var(x);
        for (int c = 0; c < std.length; c++) {
            for (int h = 0; h < std[c].length; h++) {
                for (int w = 0; w < std[c][h].length; w++) {
                    std[c][h][w] = (float) Math.sqrt(std[c][h][w]);
                }
            }
        }
        return std;
    }

    /**
     * standard deviation
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void std(float[][][][] x, float[] var, float[] std, int eta, int type) {
        //		float[] std = MatrixOperation.var(x, type);
        for (int c = 0; c < std.length; c++) {
            std[c] = (float) Math.sqrt(std[c] + eta);
        }
        //		return std;
    }

    /**
     * standard deviation
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void std(float[] x, float[] std) {
        //		float[] std = new float[x.length];
        for (int c = 0; c < std.length; c++) {
            std[c] = (float) Math.sqrt(x[c]);
        }
        //		return std;
    }

    /**
     * standard deviation
     *
     * @param x
     * @return
     */
    public static float[][][] var(float[][][][] x) {
        float[][][] mean = MatrixOperation.mean(x);
        float[][][] std = new float[x[0].length][x[0][0].length][x[0][0][0].length];
        for (int m = 0; m < x.length; m++) {
            for (int c = 0; c < x[m].length; c++) {
                for (int h = 0; h < x[m][c].length; h++) {
                    for (int w = 0; w < x[m][c][h].length; w++) {
                        std[c][h][w] += ((x[m][c][h][w] - mean[c][h][w]) * (x[m][c][h][w] - mean[c][h][w])) / x.length;
                    }
                }
            }
        }
        return std;
    }

    /**
     * standard deviation
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void var(float[][][][] x, float[] mean, float[] var, int type) {
        float scale = 1.0f / x.length;
        if (type != 0) {
            scale = 1.0f / (x.length * x[0][0].length * x[0][0][0].length);
        }
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int m = 0; m < x.length; m++) {
            final int index = m;
            final float s = scale;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    synchronized (var) {
                        for (int c = 0; c < x[index].length; c++) {
                            for (int h = 0; h < x[index][c].length; h++) {
                                for (int w = 0; w < x[index][c][h].length; w++) {
                                    if (type == 0) {
                                        var[w] += ((x[index][c][h][w] - mean[w]) * (x[index][c][h][w] - mean[w])) * s;
                                    } else {
                                        var[c] += ((x[index][c][h][w] - mean[c]) * (x[index][c][h][w] - mean[c])) * s;
                                    }
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * standard deviation
     *
     * @param x
     * @param type 0:fully,1:conv
     * @return
     */
    public static void varV2(float[][][][] x, float[] mean, float[] var, int type) {
        float scale = 1.0f / (x.length);
        if (type != 0) {
            scale = 1.0f / ((x.length * x[0][0].length * x[0][0][0].length));
        }
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        if (type == 0) {
            for (int w = 0; w < x[0][0][0].length; w++) {
                final int index = w;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int c = 0; c < x[0].length; c++) {
                            for (int h = 0; h < x[0][c].length; h++) {
                                for (int n = 0; n < x.length; n++) {
                                    val += Math.pow(x[n][c][h][index] - mean[index], 2);
                                    //									val += (x[n][c][h][index] - mean[index]) * (x[n][c][h][index] - mean[index]);
                                }
                            }
                        }
                        var[index] = val * s;
                        return null;
                    }
                });
            }
        } else {
            for (int c = 0; c < x[0].length; c++) {
                final int index = c;
                final float s = scale;
                workers.add(new Task<Object>(index) {
                    @Override
                    public Object call() throws Exception {
                        float val = 0.0f;
                        for (int n = 0; n < x.length; n++) {
                            for (int h = 0; h < x[n][index].length; h++) {
                                for (int w = 0; w < x[n][index][h].length; w++) {
                                    val += (x[n][index][h][w] - mean[index]) * (x[n][index][h][w] - mean[index]) * s;
                                }
                            }
                        }
                        var[index] = val;
                        return null;
                    }
                });
            }
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * pow
     *
     * @return
     */
    public static float[][][] pow(float[][][] x, float exponent) {
        float[][][] y = new float[x.length][x[0].length][x[0][0].length];
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    y[k][i][j] = (float) Math.pow(x[k][i][j], exponent);
                }
            }
        }
        return y;
    }

    /**
     * pow
     *
     * @return
     */
    public static float[][][][] pow(float[][][][] x, float exponent) {
        float[][][][] y = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    for (int l = 0; l < x[k][i][j].length; l++) {
                        y[k][i][j][l] = (float) Math.pow(x[k][i][j][l], exponent);
                    }
                }
            }
        }
        return y;
    }

    /**
     * sqrt
     *
     * @return
     */
    public static float[] sqrt(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) Math.sqrt(x[i]);
        }
        return y;
    }

    /**
     * sqrt
     *
     * @return
     */
    public static float[][] sqrt(float[][] x) {
        float[][] y = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                y[i][j] = (float) Math.sqrt(x[i][j]);
            }
        }
        return y;
    }

    /**
     * sqrt
     *
     * @return
     */
    public static float[][][] sqrt(float[][][] x) {
        float[][][] y = new float[x.length][x[0].length][x[0][0].length];
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    y[k][i][j] = (float) Math.sqrt(x[k][i][j]);
                }
            }
        }
        return y;
    }

    /**
     * sqrt
     *
     * @return
     */
    public static float[][][][] sqrt(float[][][][] x) {
        float[][][][] y = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
        for (int k = 0; k < x.length; k++) {
            for (int i = 0; i < x[k].length; i++) {
                for (int j = 0; j < x[k][i].length; j++) {
                    for (int l = 0; l < x[k][i][j].length; l++) {
                        y[k][i][j][l] = (float) Math.sqrt(x[k][i][j][l]);
                    }
                }
            }
        }
        return y;
    }

    /**
     * rotate90
     *
     * @param matrix
     * @return
     */
    public static float[][] rotate90(float[][] matrix) {
        float[][] temp = new float[matrix[0].length][matrix.length];
        int dst = matrix.length - 1;
        for (int i = 0; i < matrix.length; i++, dst--) {
            for (int j = 0; j < matrix[i].length; j++) {
                temp[j][dst] = matrix[i][j];
            }
        }
        return temp;
    }

    /**
     * rotate90
     *
     * @param matrix
     * @return
     */
    public static float[][][][] rotate90(float[][][][] matrix) {
        float[][][][] temp = new float[matrix.length][matrix[0].length][matrix[0][0][0].length][matrix[0][0].length];
        for (int c = 0; c < matrix.length; c++) {
            for (int k = 0; k < matrix[c].length; k++) {
                int dst = matrix[0][0].length - 1;
                for (int i = 0; i < matrix[c][k].length; i++, dst--) {
                    for (int j = 0; j < matrix[c][k][i].length; j++) {
                        temp[c][k][j][dst] = matrix[c][k][i][j];
                    }
                }
            }
        }
        return temp;
    }

    /**
     * rotate90
     *
     * @param matrix
     * @return
     */
    public static float[][][][] rotate90V2(float[][][][] matrix) {
        float[][][][] temp = new float[matrix.length][matrix[0].length][matrix[0][0][0].length][matrix[0][0].length];
        int N = matrix.length;
        int C = matrix[0].length;
        int H = matrix[0][0].length;
        int W = matrix[0][0][0].length;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int j = 0; j < matrix[n][c][0].length; j++) {
                    for (int i = matrix[n][c].length - 1; i > -1; i--) {
                        System.out.println(matrix[n][c][i][j]);
                    }
                }
            }
        }
        return temp;
    }

    public static float[][] fun180(float[][] num) {
        int m = num.length;
        int n = num[0].length;
        float[][] tmp = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                tmp[i][j] = num[m - i - 1][n - j - 1];
            }
        }
        return tmp;
    }

    public static float[][][][] rotate180V2(float[][][][] x) {
        int N = x.length;
        int C = x[0].length;
        int H = x[0][0].length;
        int W = x[0][0][0].length;
        float[][][][] tmp = new float[N][C][H][W];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        tmp[n][c][i][j] = x[n][c][H - i - 1][W - j - 1];
                    }
                }
            }
        }
        return tmp;
    }

    public static void main(String[] args) {
        float[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        PrintUtils.printImage(x);
        System.out.println("==================");
        PrintUtils.printImage(MatrixOperation.fun180(x));
    }

    /**
     * revolve180
     *
     * @param x
     * @return
     */
    public static float[][] rotate180(float[][] x) {
        float[][] temp = MatrixOperation.rotate90(x);
        temp = MatrixOperation.rotate90(temp);
        return temp;
    }

    /**
     * revolve180
     *
     * @param x
     * @return
     */
    public static float[][][][] rotate180(float[][][][] x) {
        float[][][][] temp = MatrixOperation.rotate90(x);
        temp = MatrixOperation.rotate90(temp);
        return temp;
    }

    /**
     * zeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x + padding * 2
     */
    public static float[][] zeroPadding(float[][] x, int paddingNum) {
        float[][] temp = new float[x.length + paddingNum * 2][x[0].length + paddingNum * 2];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i + paddingNum][j + paddingNum] = x[i][j];
            }
        }
        return temp;
    }

    /**
     * zeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x + padding * 2
     */
    public static float[][][] zeroPadding(float[][][] x, int paddingNum) {
        float[][][] temp = new float[x.length][x[0].length + paddingNum * 2][x[0][0].length + paddingNum * 2];
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i + paddingNum][j + paddingNum] = x[c][i][j];
                }
            }
        }
        return temp;
    }

    /**
     * zeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x + padding * 2
     */
    public static float[][][][] zeroPadding(float[][][][] x, int paddingNum) {
        float[][][][] temp = new float[x.length][x[0].length][x[0][0].length + paddingNum * 2][x[0][0][0].length + paddingNum * 2];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                temp[index][c][i + paddingNum][j + paddingNum] = x[index][c][i][j];
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return temp;
    }

    /**
     * zeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x + padding * 2
     */
    public static void zeroPadding(float[][][][] x, float[][][][] y, int paddingNum) {
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                y[index][c][i + paddingNum][j + paddingNum] = x[index][c][i][j];
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * zeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x + padding * 2
     */
    public static void zeroPadding(float[][][][] x, float[] y, int pn) {
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        int ph = x[0][0].length + 2 * pn;
        int pw = x[0][0][0].length + 2 * pn;
        for (int n = 0; n < x.length; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < x[index].length; c++) {
                        for (int i = 0; i < x[index][c].length; i++) {
                            for (int j = 0; j < x[index][c][i].length; j++) {
                                y[index * c * ph * pw + c * ph * pw + (i + pn) * pw + (j + pn)] = x[index][c][i][j];
                                //								y[index][c][i+pn][j+pn] = x[index][c][i][j];
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }

    /**
     * dislodgeZeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x - padding * 2
     */
    public static float[][] dislodgeZeroPadding(float[][] x, int paddingNum) {
        float[][] temp = new float[x.length - +paddingNum * 2][x[0].length - +paddingNum * 2];
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                temp[i][j] = x[i + paddingNum][j + paddingNum];
            }
        }
        return temp;
    }

    /**
     * dislodgeZeroPadding
     *
     * @param x
     * @param paddingNum
     * @return x - padding * 2
     */
    public static float[][][] dislodgeZeroPadding(float[][][] x, int paddingNum) {
        float[][][] temp = new float[x.length][x[0].length - paddingNum * 2][x[0][0].length - paddingNum * 2];
        for (int c = 0; c < temp.length; c++) {
            for (int i = 0; i < temp[c].length; i++) {
                for (int j = 0; j < temp[c][i].length; j++) {
                    temp[c][i][j] = x[c][i + paddingNum][j + paddingNum];
                }
            }
        }
        return temp;
    }

    /**
     * convn full
     *
     * @param x
     * @param k
     * @return zeroPadding = (K - 1) / 2
     * <p>
     * o = (W -K + 2P) / S + 1
     */
    public static float[][] convnFull(float[][] x, float[][] k, int stride) {
        int zeroPadding = k.length - 1;
        int outputSize = ((x.length - k.length + 2 * zeroPadding) / stride) + 1;
        float[][] result = new float[outputSize][outputSize];
        int kernelSize = k.length - 1;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                result[i][j] = 0.0f;
                for (int m = 0; m < k.length; m++) {
                    for (int n = 0; n < k[m].length; n++) {
                        if ((i - kernelSize + m) < 0) {
                            result[i][j] += 0.0d;
                        } else if ((j - kernelSize + n) < 0) {
                            result[i][j] += 0.0d;
                        } else if ((i - kernelSize + m) >= x.length) {
                            result[i][j] += 0.0d;
                        } else if ((j - kernelSize + n) >= x.length) {
                            result[i][j] += 0.0d;
                        } else {
                            result[i][j] += x[i - kernelSize + m][j - kernelSize + n] * k[m][n];
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * convn full
     *
     * @param x
     * @param k
     * @return zeroPadding = (K - 1) / 2
     * <p>
     * o = (W -K + 2P) / S + 1
     */
    public static float[][] convnSame(float[][] x, float[][] k, int stride) {
        int zeroPadding = (k.length - 1) / 2;
        int outputSize = ((x.length - k.length + 2 * zeroPadding) / stride) + 1;
        float[][] result = new float[outputSize][outputSize];
        int kernelSize = ((k.length + 1) / 2) - 1;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                result[i][j] = 0.0f;
                for (int m = 0; m < k.length; m++) {
                    for (int n = 0; n < k[m].length; n++) {
                        if ((i - kernelSize + m) < 0) {
                            result[i][j] += 0.0d;
                        } else if ((j - kernelSize + n) < 0) {
                            result[i][j] += 0.0d;
                        } else if ((i - kernelSize + m) >= outputSize) {
                            result[i][j] += 0.0d;
                        } else if ((j - kernelSize + n) >= outputSize) {
                            result[i][j] += 0.0d;
                        } else {
                            result[i][j] += x[i - kernelSize + m][j - kernelSize + n] * k[m][n];
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * convnVail
     *
     * @param x      h * w
     * @param k      kh * kw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][] convnVail(float[][] x, float[][] k, int stride) {
        int oHeight = ((x.length - k.length) / stride) + 1;
        int oWidth = ((x[0].length - k[0].length) / stride) + 1;
        float[][] result = new float[oHeight][oWidth];
        for (int i = 0; i < oHeight; i++) {
            for (int j = 0; j < oWidth; j++) {
                for (int m = 0; m < k.length; m++) {
                    for (int n = 0; n < k[m].length; n++) {
                        result[i][j] += x[i * stride + m][j * stride + n] * k[m][n];
                    }
                }
            }
        }
        return result;
    }

    /**
     * convnVail
     *
     * @param x      c * h * w
     * @param k      c * kn * kh * kw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][][] convnVail(float[][][] x, float[][][][] k, int stride) {
        int channel = x.length;
        int kNum = k[0].length;
        int oHeight = ((x[0].length - k[0][0].length) / stride) + 1;
        int oWidth = ((x[0][0].length - k[0][0][0].length) / stride) + 1;
        float[][][] result = new float[kNum][oHeight][oWidth];
        for (int kn = 0; kn < kNum; kn++) {
            for (int i = 0; i < oHeight; i++) {
                for (int j = 0; j < oWidth; j++) {
                    for (int c = 0; c < channel; c++) {
                        for (int m = 0; m < k[c][kn].length; m++) {
                            for (int n = 0; n < k[c][kn][m].length; n++) {
                                result[kn][i][j] += x[c][i * stride + m][j * stride + n] * k[c][kn][m][n];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * im2col2
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] im2col2(float[][][] x, int kh, int kw, int stride) {
        int oHeight = ((x[0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0].length - kw) / stride) + 1;
        int ow = x.length * kh * kw;
        int oh = oHeight * oWidth;
        int kSize = kh * kw;
        float[][] result = new float[oh][ow];
        for (int i = 0; i < oh; i++) {
            int startH = i / oHeight * stride;
            int startW = i % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int c = j / kSize;
                int xSize = j - (c * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                result[i][j] = x[c][xh][xw];
            }
        }
        return result;
    }

    /**
     * im2col2
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] im2col4d(float[][][][] x, int kh, int kw, int stride) {
        int N = x.length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int ow = x[0].length * kh * kw;
        int oh = N * oHeight * oWidth;
        int kSize = kh * kw;
        float[][] result = new float[oh][ow];
        for (int i = 0; i < oh; i++) {
            int n = i / oHeight / oWidth;
            int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
            int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int c = j / kSize;
                int xSize = j - (c * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                result[i][j] = x[n][c][xh][xw];
            }
        }
        return result;
    }

    /**
     * im2col2
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[] im2col4d(float[] x, int N, int C, int H, int W, int kh, int kw, int stride) {
        int oHeight = ((H - kh) / stride) + 1;
        int oWidth = ((W - kw) / stride) + 1;
        int ow = C * kh * kw;
        int oh = N * oHeight * oWidth;
        int kSize = kh * kw;
        float[] result = new float[oh * ow];
        for (int i = 0; i < oh; i++) {
            //			System.out.println(i);
            int n = i / oHeight / oWidth;
            int startH = (i - (n * oHeight * oWidth)) / oHeight * stride;
            int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int c = j / kSize;
                int xSize = j - (c * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                int xi = n * C * H * W + c * H * W + xh * W + xw;
                //				System.out.println(xi);
                result[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
            }
        }
        return result;
    }

    /**
     * im2col2
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] im2col4d2(float[][][][] x, int kh, int kw, int stride) {
        int N = x.length;
        int C = x[0].length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int ow = N * kh * kw;
        int oh = C * oHeight * oWidth;
        int kSize = kh * kw;
        float[][] result = new float[oh][ow];
        for (int i = 0; i < oh; i++) {
            int c = i / oHeight / oWidth;
            int startH = (i - (c * oHeight * oWidth)) / oHeight * stride;
            int startW = (i - (c * oHeight * oWidth)) % oWidth * stride;
            for (int j = 0; j < ow; j++) {
                int n = j / kSize;
                int xSize = j - (n * kSize);
                int xh = startH + xSize / kw;
                int xw = startW + xSize % kw;
                result[i][j] = x[n][c][xh][xw];
            }
        }
        return result;
    }

    /**
     * im2col
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] im2col(float[][][] x, int kh, int kw, int stride) {
        int oHeight = ((x[0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0].length - kw) / stride) + 1;
        int channel = x.length;
        float[][] result = new float[oHeight * oWidth][channel * kh * kw];
        int hIndex = 0;
        for (int oh = 0; oh < oHeight; oh++) {
            for (int ow = 0; ow < oWidth; ow++) {
                int wIndex = 0;
                for (int c = 0; c < channel; c++) {
                    for (int h = 0; h < kh; h++) {
                        for (int w = 0; w < kw; w++) {
                            result[hIndex][wIndex] = x[c][oh * stride + h][ow * stride + w];
                            wIndex++;
                        }
                    }
                }
                hIndex++;
            }
        }
        return result;
    }

    /**
     * im2col
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] im2col(float[][][][] x, int kh, int kw, int stride) {
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        int number = x.length;
        int channel = x[0].length;
        float[][] result = new float[number * oHeight * oWidth][channel * kh * kw];
        for (int n = 0; n < number; n++) {
            int hIndex = 0;
            for (int oh = 0; oh < oHeight; oh++) {
                for (int ow = 0; ow < oWidth; ow++) {
                    int wIndex = 0;
                    for (int c = 0; c < channel; c++) {
                        for (int h = 0; h < kh; h++) {
                            for (int w = 0; w < kw; w++) {
                                result[n * oHeight * oWidth + hIndex][wIndex] = x[n][c][oh * stride + h][ow * stride + w];
                                wIndex++;
                            }
                        }
                    }
                    hIndex++;
                }
            }
        }
        return result;
    }

    /**
     * im2col
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] kernel2col(float[][][][] k, int kh, int kw) {
        int ic = k.length;
        int oc = k[0].length;
        float[][] result = new float[ic * kh * kw][oc];
        int hIndex = 0;
        for (int c = 0; c < ic; c++) {
            for (int h = 0; h < kh; h++) {
                for (int w = 0; w < kw; w++) {
                    for (int n = 0; n < oc; n++) {
                        result[hIndex][n] = k[c][n][h][w];
                    }
                    hIndex++;
                }
            }
        }
        return result;
    }

    /**
     * im2col
     *
     * @param x
     * @param kh
     * @param kw
     * @param stride
     * @return
     */
    public static float[][] kernel2colToBack(float[][][][] k, int kh, int kw) {
        int ic = k.length;
        int oc = k[0].length;
        float[][] result = new float[oc * kh * kw][ic];
        int hIndex = 0;
        for (int n = 0; n < oc; n++) {
            for (int h = 0; h < kh; h++) {
                for (int w = 0; w < kw; w++) {
                    for (int c = 0; c < ic; c++) {
                        result[hIndex][c] = k[c][n][h][w];
                    }
                    hIndex++;
                }
            }
        }
        return result;
    }

    //	public static float[][][][] convnVailByIm2Col(float[][][][] x,float[][][][] k,int stride,boolean isBack){
    //
    //		int ko = k[0].length;
    //
    //		if(isBack) {
    //			ko = k.length;
    //		}
    //
    //		int kh = k[0][0].length;
    //
    //		int kw = k[0][0][0].length;
    //
    //		int oHeight = ((x[0][0].length - k[0][0].length ) / stride) + 1;
    //
    //		int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;
    //
    //		float[][][][] result = new float[x.length][ko][oHeight][oWidth];
    //
    //		Vector<Task<Object>> workers = new Vector<Task<Object>>();
    //
    //		for(int bn = 0;bn<x.length;bn++) {
    //
    //			final int index = bn;
    //
    //			workers.add(new Task<Object>(index) {
    //				@Override
    //			    public Object call() throws Exception {
    //
    //					float[][] col = MatrixOperation.im2col(x[index], kh, kw, stride);
    //
    //					float[][] colK = MatrixOperation.kernel2col(k, kh, kw);
    //
    //					if(isBack) {
    //						colK = MatrixOperation.kernel2colToBack(k, kh, kw);
    //					}
    //
    ////					float[][] output = MatrixOperation.multiplicationForMatrix(col, colK);
    //
    //					float[][] output = MatrixOperation.multiplicationByEjml(col, colK);
    //
    ////					float[][] output = MatrixOperation.multiplicationByCuda(col, colK);
    //
    //					result[index] = MatrixUtils.transform(output, oHeight, oWidth);
    //
    //					return null;
    //				}
    //			});
    //
    //		}
    //
    //		TaskEngine.getInstance(threadNum).dispatchTask(workers);
    //
    //		return result;
    //	}
    public static float[][][][] convnVailByIm2ColGPU(float[][][][] x, float[][][][] k, int stride) {
        int ko = k.length;
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int N = x.length;
        int oHeight = ((x[0][0].length - kh) / stride) + 1;
        int oWidth = ((x[0][0][0].length - kw) / stride) + 1;
        //		System.out.println(x.length+":"+x[0].length+":"+x[0][0].length+":"+x[0][0][0].length);
        long start = System.nanoTime();
        /**
         * input im2col
         *
         */
        float[][] input2d = Im2colUtils.im2col(x, kh, kw, stride);
        //		System.out.println(input2d.length+":"+input2d[0].length);
        //		long start = System.nanoTime();
        /**
         * kernel im2col
         *
         */
        float[][] kt = MatrixUtils.transpose(Im2colUtils.im2colKernel(k));
        float[] r = new float[N * ko * oHeight * oWidth];
        GPUOP.getInstance().multiplyFloat(input2d.length, kt.length, kt[0].length, MatrixUtils.transform(input2d), MatrixUtils.transform(kt), r);
        float[][][][] tmp = MatrixUtils.col2imgV2(r, N, ko, oHeight, oWidth);
        System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
        return tmp;
    }

    public static void convnVailByIm2ColGPUV2(float[] input1d, float[][][][] k, int N, int C, int H, int W, int stride, float[][][][] out) {
        //		long start = System.nanoTime();
        int ko = k.length;
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int oHeight = ((H - kh) / stride) + 1;
        int oWidth = ((W - kw) / stride) + 1;
        int xm = N * oHeight * oWidth;
        int xn = kh * kw * C;
        /**
         * kernel im2col
         *
         */
        float[] ka = Im2colUtils.kernalToVector(k, false);
        float[] r = new float[xm * ko];
        GPUOP.getInstance().multiplyFloat(xm, xn, ko, input1d, ka, r);
        MatrixUtils.col2imgV2(r, out, N, ko, oHeight, oWidth);
    }

    public static float[][][][] convnDeltaByIm2ColGPUV2(float[][][][] d, float[][][][] k, int stride) {
        //		long start = System.nanoTime();
        int kn = k.length;
        int kc = k[0].length;
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int N = d.length;
        int oHeight = ((d[0][0].length - kh) / stride) + 1;
        int oWidth = ((d[0][0][0].length - kw) / stride) + 1;
        int xm = N * oHeight * oWidth;
        int xn = kh * kw * kn;
        //		long start = System.nanoTime();
        /**
         * input im2col
         *
         */
        float[] input1d = Im2colToVector.im2col(d, kh, kw, stride);
        //		System.out.println((System.nanoTime() - start) / 1e6);
        /**
         * kernel im2col
         *
         */
        float[] kt = Im2colUtils.kernalToVector(k, true);
        float[] r = new float[N * kc * oHeight * oWidth];
        GPUOP.getInstance().multiplyFloat(xm, xn, kc, input1d, kt, r);
        float[][][][] tmp = MatrixUtils.col2imgV2(r, N, kc, oHeight, oWidth);
        //
        //		System.out.println((System.nanoTime() - start) / 1e6+"ms.");
        return tmp;
    }

    public static void convnDeltaByIm2ColGPUV2(float[][][][] d, float[][][][] k, float[][][][] diff, int stride) {
        //		long start = System.nanoTime();
        int kn = k.length;
        int kc = k[0].length;
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int N = d.length;
        int oHeight = ((d[0][0].length - kh) / stride) + 1;
        int oWidth = ((d[0][0][0].length - kw) / stride) + 1;
        int xm = N * oHeight * oWidth;
        int xn = kh * kw * kn;
        /**
         * input im2col
         *
         */
        float[] input1d = Im2colToVector.im2col(d, kh, kw, stride);
        //		System.out.println((System.nanoTime() - start) / 1e6);
        /**
         * kernel im2col
         *
         */
        float[] kt = Im2colUtils.kernalToVector(k, true);
        float[] r = new float[N * kc * oHeight * oWidth];
        GPUOP.getInstance().multiplyFloat(xm, xn, kc, input1d, kt, r);
        OP1dto4d.to1d(r, diff, N, kc, oHeight, oWidth);
        //		System.out.println("++++++++++++++++++++++++++++++++++");
        //
        //		PrintUtils.printImage(diff[2][0]);
    }

    public static float[][][][] convnDeltaByIm2ColGPU(float[][][][] d, float[][][][] k, int stride) {
        int kc = k[0].length;
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int N = d.length;
        int oHeight = ((d[0][0].length - kh) / stride) + 1;
        int oWidth = ((d[0][0][0].length - kw) / stride) + 1;
        //		long start = System.nanoTime();
        /**
         * input im2col
         *
         */
        float[][] input2d = Im2colUtils.im2col(d, kh, kw, stride);
        //		System.out.println((System.nanoTime() - start) / 1e6);
        /**
         * kernel im2col
         *
         */
        float[][] kt = Im2colUtils.kernalTo2d(k);
        float[] r = new float[N * kc * oHeight * oWidth];
        GPUOP.getInstance().multiplyFloat(input2d.length, kt.length, kt[0].length, MatrixUtils.transform(input2d), MatrixUtils.transform(kt), r);
        float[][][][] tmp = MatrixUtils.col2imgV2(r, N, kc, oHeight, oWidth);
        return tmp;
    }

    public static float[][][] convnVailByIm2Col(float[][][] x, float[][][][] k, int stride, boolean isBack) {
        int ko = k[0].length;
        if (isBack) {
            ko = k.length;
        }
        int kh = k[0][0].length;
        int kw = k[0][0][0].length;
        int oHeight = ((x[0].length - k[0][0].length) / stride) + 1;
        int oWidth = ((x[0][0].length - k[0][0][0].length) / stride) + 1;
        float[][][] result = new float[ko][oHeight][oWidth];
        float[][] col = MatrixOperation.im2col(x, kh, kw, stride);
        float[][] colK = MatrixOperation.kernel2col(k, kh, kw);
        if (isBack) {
            colK = MatrixOperation.kernel2colToBack(k, kh, kw);
        }
        float[][] output = MatrixOperation.multiplicationForMatrix(col, colK);
        result = MatrixUtils.transform(output, oHeight, oWidth);
        return result;
    }

    /**
     * convnVail
     *
     * @param x      c * h * w
     * @param k      c * kn * kh * kw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][][][] convnVail(float[][][][] x, float[][][][] k, int stride) {
        int channel = x[0].length;
        int kNum = k[0].length;
        int oHeight = ((x[0][0].length - k[0][0].length) / stride) + 1;
        int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;
        float[][][][] result = new float[x.length][kNum][oHeight][oWidth];
        for (int b = 0; b < x.length; b++) {
            float[][][] bo = new float[kNum][oHeight][oWidth];
            for (int kn = 0; kn < kNum; kn++) {
                for (int c = 0; c < channel; c++) {
                    for (int i = 0; i < oHeight; i++) {
                        for (int j = 0; j < oWidth; j++) {
                            for (int m = 0; m < k[c][kn].length; m++) {
                                for (int n = 0; n < k[c][kn][m].length; n++) {
                                    bo[kn][i][j] += x[b][c][i * stride + m][j * stride + n] * k[c][kn][m][n];
                                }
                            }
                        }
                    }
                }
            }
            result[b] = bo;
        }
        return result;
    }

    /**
     * convnVailForBack
     *
     * @param x      c * h * w
     * @param k      c * kn * kh * kw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][][][] convnVailForBack(float[][][][] k, float[][][][] x, int stride) {
        int channel = k.length;
        int kNum = k[0].length;
        int oHeight = ((x[0][0].length - k[0][0].length) / stride) + 1;
        int oWidth = ((x[0][0][0].length - k[0][0][0].length) / stride) + 1;
        float[][][][] diff = new float[x.length][channel][oHeight][oWidth];
        for (int b = 0; b < x.length; b++) {
            for (int c = 0; c < channel; c++) {
                for (int kn = 0; kn < kNum; kn++) {
                    for (int i = 0; i < oHeight; i++) {
                        for (int j = 0; j < oWidth; j++) {
                            for (int m = 0; m < k[c][kn].length; m++) {
                                for (int n = 0; n < k[c][kn][m].length; n++) {
                                    diff[b][c][i][j] += x[b][kn][i * stride + m][j * stride + n] * k[c][kn][m][n];
                                }
                            }
                        }
                    }
                }
            }
        }
        return diff;
    }

    /**
     * convnVail
     *
     * @param x      c * h * w
     * @param k      kn * kc * kh * kw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][][][] convnVail(float[][][] x, float[][][] d, int stride) {
        int channel = x.length;
        int kNum = d.length;
        int oHeight = ((x[0].length - d[0].length) / stride) + 1;
        int oWidth = ((x[0][0].length - d[0][0].length) / stride) + 1;
        float[][][][] result = new float[channel][kNum][oHeight][oWidth];
        for (int c = 0; c < channel; c++) {
            for (int kn = 0; kn < kNum; kn++) {
                for (int i = 0; i < oHeight; i++) {
                    for (int j = 0; j < oWidth; j++) {
                        for (int m = 0; m < d[kn].length; m++) {
                            for (int n = 0; n < d[kn][m].length; n++) {
                                result[c][kn][i][j] += x[c][i * stride + m][j * stride + n] * d[kn][m][n];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * convnVailForDelta
     *
     * @param x      n * c * h * w
     * @param d      n * do * dh * dw
     * @param stride
     * @return o = (W - K) / S + 1
     */
    public static float[][][][] convnVailForDelta(float[][][][] x, float[][][][] d, int stride) {
        int channel = x[0].length;
        int kNum = d[0].length;
        int oHeight = ((x[0][0].length - d[0][0].length) / stride) + 1;
        int oWidth = ((x[0][0][0].length - d[0][0][0].length) / stride) + 1;
        float[][][][] result = new float[channel][kNum][oHeight][oWidth];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int b = 0; b < x.length; b++) {
            int index = b;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < channel; c++) {
                        for (int kn = 0; kn < kNum; kn++) {
                            for (int i = 0; i < oHeight; i++) {
                                for (int j = 0; j < oWidth; j++) {
                                    for (int m = 0; m < d[index][kn].length; m++) {
                                        for (int n = 0; n < d[index][kn][m].length; n++) {
                                            result[c][kn][i][j] += x[index][c][i * stride + m][j * stride + n] * d[index][kn][m][n];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return result;
    }

    /**
     * pooling
     *
     * @param x
     * @param pWidth
     * @param pHeight
     * @param stride
     * @param poolingType
     * @return o = (W - ps) / S + 1
     */
    public static float[][][] pooling(float[][][] x, int pWidth, int pHeight, int stride, PoolingType poolingType) {
        int channel = x.length;
        int oHeight = ((x[0].length - pHeight) / stride) + 1;
        int oWidth = ((x[0][0].length - pWidth) / stride) + 1;
        float[][][] result = new float[channel][oHeight][oWidth];
        for (int c = 0; c < channel; c++) {
            for (int i = 0; i < oHeight; i++) {
                for (int j = 0; j < oWidth; j++) {
                    for (int m = 0; m < pHeight; m++) {
                        for (int n = 0; n < pWidth; n++) {
                            switch (poolingType) {
                                case MAX_POOLING:
                                    if (m == 0 && n == 0) {
                                        result[c][i][j] = x[c][i * stride + m][j * stride + n];
                                    } else if (result[c][i][j] < x[c][i * stride + m][j * stride + n]) {
                                        result[c][i][j] = x[c][i * stride + m][j * stride + n];
                                    }
                                    break;
                                case MEAN_POOLING:
                                    result[c][i][j] += x[c][i * stride + m][j * stride + n];
                                    break;
                            }
                        }
                    }
                    switch (poolingType) {
                        case MAX_POOLING:
                            break;
                        case MEAN_POOLING:
                            result[c][i][j] /= pWidth * pHeight;
                            break;
                    }
                }
            }
        }
        return result;
    }

    /**
     * pooling
     *
     * @param x
     * @param pWidth
     * @param pHeight
     * @param stride
     * @param poolingType
     * @return o = (W - ps) / S + 1
     */
    public static float[][][] poolingAndMask(float[][][] x, float[][][][] mask, int pWidth, int pHeight, int stride, PoolingType poolingType) {
        int channel = x.length;
        int oHeight = ((x[0].length - pHeight) / stride) + 1;
        int oWidth = ((x[0][0].length - pWidth) / stride) + 1;
        float[][][] result = new float[channel][oHeight][oWidth];
        for (int c = 0; c < channel; c++) {
            int maskIndex = 0;
            for (int i = 0; i < oHeight; i++) {
                for (int j = 0; j < oWidth; j++) {
                    int maxH = 0;
                    int maxW = 0;
                    for (int m = 0; m < pHeight; m++) {
                        for (int n = 0; n < pWidth; n++) {
                            switch (poolingType) {
                                case MAX_POOLING:
                                    if (m == 0 && n == 0) {
                                        result[c][i][j] = x[c][i * stride + m][j * stride + n];
                                    } else if (result[c][i][j] <= x[c][i * stride + m][j * stride + n]) {
                                        result[c][i][j] = x[c][i * stride + m][j * stride + n];
                                        maxH = m;
                                        maxW = n;
                                    }
                                    break;
                                case MEAN_POOLING:
                                    result[c][i][j] += x[c][i * stride + m][j * stride + n];
                                    mask[c][maskIndex][m][n] = 1.0f / pWidth / pHeight;
                                    break;
                            }
                        }
                    }
                    switch (poolingType) {
                        case MAX_POOLING:
                            mask[c][maskIndex][maxH][maxW] = 1;
                            break;
                        case MEAN_POOLING:
                            result[c][i][j] /= pWidth * pHeight;
                            break;
                    }
                    maskIndex++;
                }
            }
        }
        return result;
    }

    /**
     * pooling
     *
     * @param x
     * @param pWidth
     * @param pHeight
     * @param stride
     * @param poolingType
     * @return o = (W - ps) / S + 1
     */
    public static float[][][][] poolingAndMask(float[][][][] x, float[][][][][] mask, int pWidth, int pHeight, int stride, PoolingType poolingType) {
        int number = x.length;
        int channel = x[0].length;
        int oHeight = ((x[0][0].length - pHeight) / stride) + 1;
        int oWidth = ((x[0][0][0].length - pWidth) / stride) + 1;
        float[][][][] result = new float[number][channel][oHeight][oWidth];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int b = 0; b < number; b++) {
            final int index = b;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() {
                    for (int c = 0; c < channel; c++) {
                        int maskIndex = 0;
                        for (int i = 0; i < oHeight; i++) {
                            for (int j = 0; j < oWidth; j++) {
                                int maxH = 0;
                                int maxW = 0;
                                for (int m = 0; m < pHeight; m++) {
                                    for (int n = 0; n < pWidth; n++) {
                                        switch (poolingType) {
                                            case MAX_POOLING:
                                                if (m == 0 && n == 0) {
                                                    result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
                                                } else if (result[index][c][i][j] <= x[index][c][i * stride + m][j * stride + n]) {
                                                    result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
                                                    maxH = m;
                                                    maxW = n;
                                                }
                                                break;
                                            case MEAN_POOLING:
                                                result[index][c][i][j] += x[index][c][i * stride + m][j * stride + n];
                                                mask[index][c][maskIndex][m][n] = 1.0f / pWidth / pHeight;
                                                break;
                                        }
                                    }
                                }
                                switch (poolingType) {
                                    case MAX_POOLING:
                                        mask[index][c][maskIndex][maxH][maxW] = 1;
                                        break;
                                    case MEAN_POOLING:
                                        result[index][c][i][j] /= pWidth * pHeight;
                                        break;
                                }
                                maskIndex++;
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return result;
    }

    /**
     * pooling
     *
     * @param x
     * @param pWidth
     * @param pHeight
     * @param stride
     * @param poolingType
     * @return o = (W - ps) / S + 1
     */
    public static void poolingAndMask(float[][][][] x, float[][][][][] mask, int pWidth, int pHeight, int stride, PoolingType poolingType, float[][][][] result) {
        int number = x.length;
        int channel = x[0].length;
        int oHeight = ((x[0][0].length - pHeight) / stride) + 1;
        int oWidth = ((x[0][0][0].length - pWidth) / stride) + 1;
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int b = 0; b < number; b++) {
            final int index = b;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() {
                    for (int c = 0; c < channel; c++) {
                        int maskIndex = 0;
                        for (int i = 0; i < oHeight; i++) {
                            for (int j = 0; j < oWidth; j++) {
                                int maxH = 0;
                                int maxW = 0;
                                float maxval = -3.402823466e+38f;
                                float meanVal = 0.0f;
                                for (int m = 0; m < pHeight; m++) {
                                    for (int n = 0; n < pWidth; n++) {
                                        switch (poolingType) {
                                            case MAX_POOLING:
                                                float val = x[index][c][i * stride + m][j * stride + n];
                                                if (maxval <= val) {
                                                    maxH = m;
                                                    maxW = n;
                                                    maxval = val;
                                                }
                                                break;
                                            case MEAN_POOLING:
                                                meanVal += x[index][c][i * stride + m][j * stride + n];
                                                mask[index][c][maskIndex][m][n] = 1.0f / (pWidth * pHeight);
                                                break;
                                        }
                                    }
                                }
                                switch (poolingType) {
                                    case MAX_POOLING:
                                        result[index][c][i][j] = maxval;
                                        mask[index][c][maskIndex][maxH][maxW] = 1;
                                        break;
                                    case MEAN_POOLING:
                                        result[index][c][i][j] = meanVal / (pWidth * pHeight);
                                        break;
                                }
                                maskIndex++;
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
    }
    //	/**
    //	 * pooling
    //	 * @param x
    //	 * @param pWidth
    //	 * @param pHeight
    //	 * @param stride
    //	 * @param poolingType
    //	 * @return
    //	 *
    //	 * o = (W - ps) / S + 1
    //	 */
    //	public static void poolingAndMask(float[] x,float[] mask,int pWidth,int pHeight,int stride,PoolingType poolingType,float[] result){
    //
    //		int number = x.length;
    //
    //		int channel = x[0].length;
    //
    //		int oHeight = ((x[0][0].length - pHeight ) / stride) + 1;
    //
    //		int oWidth = ((x[0][0][0].length - pWidth) / stride) + 1;
    //
    //		Vector<Task<Object>> workers = new Vector<Task<Object>>();
    //
    //		for(int b = 0;b<number;b++) {
    //
    //			final int index = b;
    //
    //			workers.add(new Task<Object>(index) {
    //
    //				@Override
    //				public Object call() {
    //
    //					for(int c = 0;c<channel;c++) {
    //
    //						int maskIndex = 0;
    //
    //						for(int i = 0;i<oHeight;i++) {
    //
    //							for(int j = 0;j<oWidth;j++) {
    //
    //								int maxH = 0;
    //
    //								int maxW = 0;
    //
    //								for(int m = 0;m<pHeight;m++) {
    //
    //									for(int n = 0;n<pWidth;n++) {
    //
    //										switch (poolingType) {
    //										case MAX_POOLING:
    //
    //											if(m == 0 && n == 0) {
    //												result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
    //											}else if(result[index][c][i][j] <= x[index][c][i * stride + m][j * stride + n]) {
    //												result[index][c][i][j] = x[index][c][i * stride + m][j * stride + n];
    //												maxH = m;
    //												maxW = n;
    //											}
    //
    //											break;
    //										case MEAN_POOLING:
    //											result[index][c][i][j] += x[index][c][i * stride + m][j * stride + n];
    //											mask[index][c][maskIndex][m][n] = 1.0f / pWidth / pHeight;
    //											break;
    //										}
    //
    //									}
    //
    //								}
    //
    //								switch (poolingType) {
    //								case MAX_POOLING:
    //									mask[index][c][maskIndex][maxH][maxW] = 1;
    //									break;
    //								case MEAN_POOLING:
    //									result[index][c][i][j] /= pWidth * pHeight;
    //									break;
    //								}
    //
    //								maskIndex++;
    //							}
    //
    //						}
    //
    //					}
    //
    //					return null;
    //				}
    //
    //			});
    //
    //		}
    //
    //		TaskEngine.getInstance(threadNum).dispatchTask(workers);
    //
    //	}

    /**
     * poolingDiff
     *
     * @param delta
     * @param mask
     * @return
     */
    public static float[][][] poolingDiff(float[][][] delta, float[][][][] mask, float[][][] diff, int pWidth, int pHeight, int stride) {
        int channel = diff.length;
        int oHeight = ((diff[0].length - pHeight) / stride) + 1;
        int oWidth = ((diff[0][0].length - pWidth) / stride) + 1;
        for (int c = 0; c < channel; c++) {
            int maskIndex = 0;
            for (int i = 0; i < oHeight; i++) {
                for (int j = 0; j < oWidth; j++) {
                    for (int m = 0; m < pHeight; m++) {
                        for (int n = 0; n < pWidth; n++) {
                            diff[c][i * stride + m][j * stride + n] += delta[c][i][j] * mask[c][maskIndex][m][n];
                        }
                    }
                    maskIndex++;
                }
            }
        }
        return diff;
    }

    /**
     * poolingDiff
     *
     * @param delta
     * @param mask
     * @return
     */
    public static float[][][][] poolingDiff(float[][][][] delta, float[][][][][] mask, float[][][][] diff, int pWidth, int pHeight, int stride) {
        int number = diff.length;
        int channel = diff[0].length;
        int oHeight = ((diff[0][0].length - pHeight) / stride) + 1;
        int oWidth = ((diff[0][0][0].length - pWidth) / stride) + 1;
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int b = 0; b < number; b++) {
            final int index = b;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    synchronized (diff) {
                        for (int c = 0; c < channel; c++) {
                            int maskIndex = 0;
                            for (int i = 0; i < oHeight; i++) {
                                for (int j = 0; j < oWidth; j++) {
                                    for (int m = 0; m < pHeight; m++) {
                                        for (int n = 0; n < pWidth; n++) {
                                            diff[index][c][i * stride + m][j * stride + n] = delta[index][c][i][j] * mask[index][c][maskIndex][m][n];
                                        }
                                    }
                                    maskIndex++;
                                }
                            }
                        }
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return diff;
    }
}
