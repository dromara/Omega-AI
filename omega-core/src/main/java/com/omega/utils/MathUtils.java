package com.omega.utils;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 数学工具
 *
 * @author Administrator
 */
public class MathUtils {
    public static Random random;

    /**
     * 生成随机数组
     *
     * @param length
     * @return
     */
    public static int[][] orderInts(int length, int batchSize) {
        int itc = new BigDecimal(length).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        int[][] tmp = new int[itc][batchSize];
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < length; i++) {
            list.add(i);
        }
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= length) {
                    tmp[i][j] = list.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = list.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    /**
     * 生成随机数
     *
     * @param length
     * @return
     */
    public static Integer[] randomInts(int length) {
        Integer[] tmp = new Integer[length];
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < length; i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        tmp = list.toArray(tmp);
        return tmp;
    }

    /**
     * 生成随机数组
     *
     * @param length
     * @return
     */
    public static int[][] randomInts(int length, int batchSize) {
        int itc = new BigDecimal(length).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        int[][] tmp = new int[itc][batchSize];
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < length; i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= length) {
                    tmp[i][j] = list.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = list.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    /**
     * 生成随机数组
     *
     * @param length
     * @return
     */
    public static int[][] randomInts(int length, int batchSize, int[][] tmp, List<Integer> list) {
        list.clear();
        for (int i = 0; i < length; i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= length) {
                    tmp[i][j] = list.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = list.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    public static void main(String[] args) {
        int length = 1000;
        int batchSize = 64;
        System.out.println(JsonUtils.toJson(randomInts(length, batchSize)));
    }

    /**
     * 生成随机数组
     *
     * @param length
     * @return
     */
    public static int[][] sortInts(int length, int batchSize) {
        int itc = new BigDecimal(length).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
        int[][] tmp = new int[itc][batchSize];
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < length; i++) {
            list.add(i);
        }
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (i * batchSize + j >= length) {
                    tmp[i][j] = list.get(0 * batchSize + j);
                } else {
                    tmp[i][j] = list.get(i * batchSize + j);
                }
            }
        }
        return tmp;
    }

    /**
     * 整形随机数(范围)
     *
     * @param start
     * @param end
     * @return
     */
    public static int randomInt(int end) {
        return MathUtils.getRandom().nextInt(end);
    }

    /**
     * 整形随机数(范围)
     *
     * @param end
     * @param size
     * @return
     */
    public static int[] randomInt(int end, int size) {
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            result[i] = MathUtils.getRandom().nextInt(end);
        }
        return result;
    }

    /**
     * 整形随机数不重复(范围)
     *
     * @param end
     * @param size
     * @return
     */
    public static int[] randomIntNotRepeat(int end, int size) {
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            int temp = MathUtils.getRandom().nextInt(end);
            MathUtils.randomIntNotRepeat(result, i, temp, end);
        }
        return result;
    }

    public static void randomIntNotRepeat(int[] data, int index, int value, int end) {
        for (int i = 0; i < data.length; i++) {
            if (!MathUtils.isExist(data, value)) {
                data[index] = value;
                break;
            } else {
                int temp = MathUtils.getRandom().nextInt(end);
                MathUtils.randomIntNotRepeat(data, index, temp, end);
            }
        }
    }

    public static boolean isExist(int[] data, int value) {
        for (int temp : data) {
            if (temp == value) {
                return true;
            }
        }
        return false;
    }

    public static Random getRandom() {
        if (random == null) {
            random = new Random();
        }
        return random;
    }

    public static float max(float[] x) {
        float max = 0.0f;
        for (float t : x) {
            if (max <= t) {
                max = t;
            }
        }
        return max;
    }

    public static float min(float[] x) {
        float min = 0.0f;
        for (float t : x) {
            if (min >= t) {
                min = t;
            }
        }
        return min;
    }
}

