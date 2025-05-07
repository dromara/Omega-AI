package com.omega.common.utils;

import com.omega.common.task.ForkJobEngine;
import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;

import java.util.Vector;

/**
 * MatrixUtils
 *
 * @author Administrator
 */
public class MatrixUtils {
    private static final int threadNum = 8;

    public static float sum(float[] x) {
        float y = 0.0f;
        for (int i = 0; i < x.length; i++) {
            y += x[i];
        }
        return y;
    }

    public static boolean isZero(float[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] zero(int size) {
        return new float[size];
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void zero(int size, float[] x) {
        if (x == null) {
            x = new float[size];
        } else {
            MatrixUtils.zero(x);
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void zero(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = 0;
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] one(int size) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = 1.0f;
        }
        return temp;
    }

    public static float[] one(long size) {
        float[] temp = new float[(int) size];
        for (int i = 0; i < size; i++) {
            temp[i] = 1.0f;
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] order(int size, int tmp) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = 0.1f * i + tmp;
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static int[] orderInt(int size, int tmp) {
        int[] temp = new int[size];
        for (int i = 0; i < size; i++) {
            temp[i] = i + tmp;
        }
        return temp;
    }

    public static int[] orderRank(int start, int end) {
        int[] temp = new int[end - start];
        for (int i = 0; i < (end - start); i++) {
            temp[i] = i + start;
        }
        return temp;
    }

    public static Integer[] orderRankInt(int start, int end) {
        Integer[] temp = new Integer[end - start];
        for (int i = 0; i < (end - start); i++) {
            temp[i] = i + start;
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] order(int size, int tmp, int b) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = b * i + tmp;
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] order(int size, float tmp, float b) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = b * i + tmp;
        }
        return temp;
    }

    public static float[] linspace(float start, float end, int count) {
        float step = (end - start) / (count - 1);
        float[] output = new float[count];
        output[0] = start;
        for (int i = 1; i < count; i++) {
            if (i == count - 1) {
                output[i] = end;
            } else {
                output[i] = output[i - 1] + step;
            }
        }
        return output;
    }

    public static float[] range(float start, float end, int step, float tmp) {
        int count = (int) ((end - start) / step);
        float[] output = new float[count];
        output[0] = start + tmp;
        for (int i = 1; i < count; i++) {
            output[i] = output[i - 1] + step;
        }
        return output;
    }

    public static float[] cumprod(float[] x) {
        float[] output = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            if (i == 0) {
                output[i] = x[i];
            } else {
                output[i] = output[i - 1] * x[i];
            }
        }
        return output;
    }

    public static float[] gather(float[] v, int[] index) {
        float[] output = new float[index.length];
        for (int i = 0; i < index.length; i++) {
            output[i] = v[index[i]];
        }
        return output;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] val(int size, float v) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = v;
        }
        return temp;
    }

    public static int[] valInt(int size, int v) {
        int[] temp = new int[size];
        for (int i = 0; i < size; i++) {
            temp[i] = v;
        }
        return temp;
    }

    public static float[] val(int size, int v) {
        float[] temp = new float[size];
        for (int i = 0; i < size; i++) {
            temp[i] = v;
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void val(float[] x, float v) {
        for (int i = 0; i < x.length; i++) {
            x[i] = v;
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] createMatrix(int heigth, int width, float value) {
        float[][] temp = new float[heigth][width];
        for (int h = 0; h < heigth; h++) {
            for (int w = 0; w < width; w++) {
                temp[h][w] = value;
            }
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] createMatrix(int channel, int heigth, int width, float value) {
        float[][][] temp = new float[channel][heigth][width];
        for (int c = 0; c < channel; c++) {
            for (int h = 0; h < heigth; h++) {
                for (int w = 0; w < width; w++) {
                    temp[c][h][w] = value;
                }
            }
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] createMatrixByIndex(int channel, int heigth, int width) {
        float[][][] temp = new float[channel][heigth][width];
        for (int c = 0; c < channel; c++) {
            float index = 1;
            for (int h = 0; h < heigth; h++) {
                for (int w = 0; w < width; w++) {
                    temp[c][h][w] = index;
                    index++;
                }
            }
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] createMatrixByIndex(int number, int channel, int heigth, int width) {
        float[][][][] temp = new float[number][channel][heigth][width];
        for (int n = 0; n < number; n++) {
            for (int c = 0; c < channel; c++) {
                float index = 1;
                for (int h = 0; h < heigth; h++) {
                    for (int w = 0; w < width; w++) {
                        temp[n][c][h][w] = index;
                        index++;
                    }
                }
            }
        }
        return temp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] zero(int x, int y) {
        return new float[x][y];
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void zero(float[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = 0;
            }
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] zero(int x, int y, int z) {
        return new float[x][y][z];
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] zero(int n, int c, int h, int w) {
        return new float[n][c][h][w];
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void zero(float[][][][] x) {
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        x[n][c][h][w] = 0;
                    }
                }
            }
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][][] zero(int x, int y, int z, int n, int m) {
        return new float[x][y][z][n][m];
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void zero(float[][][][][] x) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                for (int m = 0; m < x[i][j].length; m++) {
                    for (int n = 0; n < x[i][j][m].length; n++) {
                        for (int o = 0; o < x[i][j][m][n].length; o++) {
                            x[i][j][m][n][o] = 0;
                        }
                    }
                }
            }
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: val for matrix
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] val(int x, int y, int z, int n, float val) {
        float[][][][] tmp = new float[x][y][z][n];
        for (int c = 0; c < tmp.length; c++) {
            for (int i = 0; i < tmp[c].length; i++) {
                for (int j = 0; j < tmp[c][i].length; j++) {
                    for (int l = 0; l < tmp[c][i][j].length; l++) {
                        tmp[c][i][j][l] = val;
                    }
                }
            }
        }
        return tmp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: val for matrix
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][][] val(int x, int y, int z, int n, float p, float val) {
        float[][][][] tmp = new float[x][y][z][n];
        for (int c = 0; c < tmp.length; c++) {
            for (int i = 0; i < tmp[c].length; i++) {
                for (int j = 0; j < tmp[c][i].length; j++) {
                    for (int l = 0; l < tmp[c][i][j].length; l++) {
                        if (RandomUtils.getInstance().nextFloat() > p) {
                            tmp[c][i][j][l] = val;
                        }
                    }
                }
            }
        }
        return tmp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: val for matrix
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] val(int n, float p, float val) {
        float[] tmp = new float[n];
        for (int c = 0; c < n; c++) {
            if (RandomUtils.getInstance().nextFloat() > p) {
                tmp[c] = val;
            }
        }
        return tmp;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] clear(float[] x) {
        if (x != null) {
            x = MatrixUtils.zero(x.length);
        }
        return x;
    }

    public static int countOccurrences(float[] arr, float target) {
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                count++;
            }
        }
        return count;
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static void fill(int[] x, int start, int end, int val) {
        for (int i = start; i < end; i++) {
            x[i] = val;
        }
    }

    /**
     * @param size
     * @return
     * @throws
     * @Title: zero
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] clear(float[][] x) {
        if (x != null) {
            x = MatrixUtils.zero(x.length, x[0].length);
        }
        return x;
    }

    /**
     * size
     *
     * @param data
     * @return
     */
    public static int[] size(float[][][][] data) {
        //		System.out.println("["+data.length+","+data[0].length+","+data[0][0].length+","+data[0][0][0].length+"]");
        return new int[]{data.length, data[0].length, data[0][0].length, data[0][0][0].length};
    }

    /**
     * size
     *
     * @param data
     * @return
     */
    public static int[] size(float[][] data) {
        System.out.println("[" + data.length + "," + data[0].length);
        return new int[]{data.length, data[0].length};
    }

    /**
     * @param x
     * @return
     * @throws
     * @Title: clone
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[] clone(float[] x) {
        float[] temp = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i];
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws
     * @Title: clone
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][] clone(float[][] x) {
        float[][] temp = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                temp[i][j] = x[i][j];
            }
        }
        return temp;
    }

    /**
     * @param x
     * @return
     * @throws
     * @Title: clone
     * @Description: TODO(这里用一句话描述这个方法的作用)
     */
    public static float[][][] clone(float[][][] x) {
        float[][][] temp = new float[x.length][x[0].length][x[0][0].length];
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[c].length; i++) {
                for (int j = 0; j < x[c][i].length; j++) {
                    temp[c][i][j] = x[c][i][j];
                }
            }
        }
        return temp;
    }

    /**
     * transform
     *
     * @param x c * h * w
     * @return
     * @index ci * h * w + hi * w + wi
     */
    public static float[] transform(float[][][] x) {
        float[] result = new float[x.length * x[0].length * x[0][0].length];
        for (int c = 0; c < x.length; c++) {
            for (int h = 0; h < x[c].length; h++) {
                for (int w = 0; w < x[c][h].length; w++) {
                    result[c * x[c].length * x[c][h].length + h * x[c][h].length + w] = x[c][h][w];
                }
            }
        }
        return result;
    }

    /**
     * transform
     *
     * @param x c * h * w
     * @return
     * @index ci * h * w + hi * w + wi
     */
    public static float[] transform(float[][] x) {
        int xi = x.length;
        int xj = x[0].length;
        float[] y = new float[xi * xj];
        for (int r = 0; r < xi; r++) {
            for (int c = 0; c < xj; c++) {
                y[r * xj + c] = x[r][c];
            }
        }
        return y;
    }

    /**
     * transform
     *
     * @param x c * h * w
     * @return
     * @index ci * h * w + hi * w + wi
     */
    public static void transform(float[][] x, float[] y) {
        int xi = x.length;
        int xj = x[0].length;
        for (int r = 0; r < xi; r++) {
            for (int c = 0; c < xj; c++) {
                y[r * xj + c] = x[r][c];
            }
        }
    }

    /**
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[] transform(float[][][][] x) {
        float[] y = new float[x.length * x[0].length * x[0][0].length * x[0][0][0].length];
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        y[n * x[n].length * x[n][c].length * x[n][c][h].length + c * x[n][c].length * x[n][c][h].length + h * x[n][c][h].length + w] = x[n][c][h][w];
                    }
                }
            }
        }
        return y;
    }

    /**
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static void transform(float[][][][] x, float[] y) {
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < x[n].length; c++) {
                for (int h = 0; h < x[n][c].length; h++) {
                    for (int w = 0; w < x[n][c][h].length; w++) {
                        y[n * x[n].length * x[n][c].length * x[n][c][h].length + c * x[n][c].length * x[n][c][h].length + h * x[n][c][h].length + w] = x[n][c][h][w];
                    }
                }
            }
        }
    }

    /**
     * transform
     *
     * @param x
     * @return
     */
    public static float[][] transform(float[] x, int r, int c) {
        float[][] y = new float[r][c];
        for (int ri = 0; ri < r; ri++) {
            for (int ci = 0; ci < c; ci++) {
                y[ri][ci] = x[ri * c + ci];
            }
        }
        return y;
    }

    /**
     * transform
     *
     * @param x
     * @return
     */
    public static void transform(float[] x, float[][] y, int r, int c) {
        for (int ri = 0; ri < r; ri++) {
            for (int ci = 0; ci < c; ci++) {
                y[ri][ci] = x[ri * c + ci];
            }
        }
    }

    /**
     * transform
     *
     * @param x
     * @return
     */
    public static float[][][] transform(float[] x, int c, int h, int w) {
        float[][][] y = new float[c][h][w];
        for (int ci = 0; ci < c; ci++) {
            for (int hi = 0; hi < h; hi++) {
                for (int wi = 0; wi < w; wi++) {
                    y[ci][hi][wi] = x[ci * h * w + hi * w + wi];
                }
            }
        }
        return y;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[][][][] transform(float[] x, int n, int c, int h, int w) {
        float[][][][] y = new float[n][c][h][w];
        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                for (int hi = 0; hi < h; hi++) {
                    for (int wi = 0; wi < w; wi++) {
                        y[ni][ci][hi][wi] = x[ni * c * h * w + ci * h * w + hi * w + wi];
                    }
                }
            }
        }
        return y;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static void transform(float[] x, float[][][][] y, int n, int c, int h, int w) {
        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                for (int hi = 0; hi < h; hi++) {
                    for (int wi = 0; wi < w; wi++) {
                        y[ni][ci][hi][wi] = x[ni * c * h * w + ci * h * w + hi * w + wi];
                    }
                }
            }
        }
    }

    /**
     * transform
     *
     * @param x
     * @return
     */
    public static float[][][][] transform(float[][] x, int n, int c, int h, int w) {
        float[][][][] y = new float[n][c][h][w];
        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                for (int hi = 0; hi < h; hi++) {
                    for (int wi = 0; wi < w; wi++) {
                        y[ni][ci][hi][wi] = x[ni][ci * h * w + hi * w + wi];
                    }
                }
            }
        }
        return y;
    }

    /**
     * transform
     *
     * @param x c * h * w
     * @return
     */
    public static float[][][] transform(float[][] x, int oh, int ow) {
        int channel = x[0].length;
        float[][][] result = new float[channel][oh][ow];
        for (int c = 0; c < channel; c++) {
            int index = 0;
            for (int h = 0; h < oh; h++) {
                for (int w = 0; w < ow; w++) {
                    result[c][h][w] = x[index][c];
                    index++;
                }
            }
        }
        return result;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @remark 数据长度必须对等
     */
    public static float[][][][] transform(float[][][][] x, int n, int c, int h, int w) {
        float[] temp = MatrixUtils.transform(x);
        return MatrixUtils.transform(temp, n, c, h, w);
    }

    /**
     * 矩阵转置
     *
     * @return
     */
    public static float[][] transpose(float[][] x) {
        float[][] result = new float[x[0].length][x.length];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int i = 0; i < x.length; i++) {
            final int index = i;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int j = 0; j < x[0].length; j++) {
                        result[j][index] = x[index][j];
                    }
                    return null;
                }
            });
        }
        TaskEngine.getInstance(threadNum).dispatchTask(workers);
        return result;
    }

    /**
     * 矩阵转置
     *
     * @return
     */
    public static float[] transpose(float[] x, int m, int n) {
        float[] y = new float[m * n];
        Transpose job = new Transpose(x, y, m, n, 0, (m * n - 1));
        ForkJobEngine.run(job);
        return y;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[][][][] col2img(float[] x, int N, int C, int H, int W) {
        float[][][][] result = new float[N][C][H][W];
        float[][] mat = to2DimenArray(x, N * H * W, C);
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < N; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                result[index][c][h][w] = mat[index * H * W + h * W + w][c];
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
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[][][][] col2imgV2(float[] x, int N, int C, int H, int W) {
        float[][][][] result = new float[N][C][H][W];
        float[][] mat = to2DimenArray(x, N * H * W, C);
        OP2dTo4d job = new OP2dTo4d(mat, result, 0, mat.length - 1);
        ForkJobEngine.run(job);
        return result;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[][][][] col2imgV2(float[] x, float[][][][] y, int N, int C, int H, int W) {
        float[][] mat = to2DimenArray(x, N * H * W, C);
        OP2dTo4d job = new OP2dTo4d(mat, y, 0, mat.length - 1);
        ForkJobEngine.run(job);
        return y;
    }

    /**
     * transform
     *
     * @param x
     * @return
     * @index ni * c * h * w + ci * h * w + hi * w + wi
     */
    public static float[][][][] col2img(float[][] x, int N, int C, int H, int W) {
        float[][][][] result = new float[N][C][H][W];
        Vector<Task<Object>> workers = new Vector<Task<Object>>();
        for (int n = 0; n < N; n++) {
            final int index = n;
            workers.add(new Task<Object>(index) {
                @Override
                public Object call() throws Exception {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                result[index][c][h][w] = x[index * H * W + h * W + w][c];
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

    public static float[][] to2DimenArray(float[] x, int n, int d) {
        float[][] result = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x, i * d, result[i], 0, d);
        }
        return result;
    }

    public static int[] shape(float[][][][] x) {
        return new int[]{x.length, x[0].length, x[0][0].length, x[0][0][0].length};
    }

    public static int[] shape(float[][] x) {
        return new int[]{x.length, x[0].length};
    }

    public static void col2im4d(float[] once, float[][][][] out, int index, int C, int H, int W) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    out[index][c][h][w] = once[c * H * W + h * W + w];
                }
            }
        }
    }

    public static void col2im4dWeight(float[] once, float[][][][] out, int kNumber, int channel, int H, int W, float number) {
        //		System.out.println(out);
        for (int ko = 0; ko < kNumber; ko++) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        out[ko][c][h][w] = once[ko * channel * H * W + c * H * W + h * W + w] / number;
                    }
                }
            }
        }
    }

    public static float check(float[] a, float[] b) {
        float error = 0.0f;
        int count = 0;
        for (int i = 0; i < a.length; i++) {
            float t = Math.abs(a[i] - b[i]);
            error += t;
            if (t > 0) {
                count++;
            }
        }
        System.out.println(count);
        return error;
    }
}

