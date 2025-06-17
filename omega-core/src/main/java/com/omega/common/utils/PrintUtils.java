package com.omega.common.utils;

import com.omega.engine.tensor.Tensor;

/**
 * PrintUtils
 *
 * @author Administrator
 */
public class PrintUtils {
    /**
     * print matrix
     *
     * @param data
     */
    public static void printImage(Tensor data) {
        if (data.isHasGPU()) {
            data.syncHost();
        }
        for (int n = 0; n < data.getShape()[0]; n++) {
            for (int c = 0; c < data.getShape()[1]; c++) {
                for (int i = 0; i < data.getShape()[2]; i++) {
                    for (int j = 0; j < data.getShape()[3]; j++) {
                        System.out.print(data.getData()[n * data.getOnceSize() + c * data.getHeight() * data.getWidth() + i * data.getWidth() + j] + " ");
                    }
                    System.out.println("");
                }
                System.out.println("----------------");
            }
            System.out.println("====================");
        }
    }

    public static void printImage(float[] data, int number, int channel, int height, int width) {
        int onceSize = channel * height * width;
        for (int n = 0; n < number; n++) {
            for (int c = 0; c < channel; c++) {
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        System.out.print(data[n * onceSize + c * height * width + i * width + j] + " ");
                    }
                    System.out.println("");
                }
                System.out.println("----------------");
            }
            System.out.println("====================");
        }
    }

    /**
     * print matrix
     *
     * @param data
     */
    public static void printImage(float[] data) {
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i] + " ");
        }
    }

    /**
     * print matrix
     *
     * @param data
     */
    public static void printImage(float[] data, int w, int h) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                System.out.print(data[i * w + j] + " ");
            }
            System.out.println("");
        }
    }

    /**
     * print matrix
     *
     * @param data
     */
    public static void printImage(float[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                System.out.print(data[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("-----------------------------------");
    }

    /**
     * print matrix
     *
     * @param data
     */
    public static void printImage(int[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                System.out.print(data[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("-----------------------------------");
    }

    public static void printImage(float[][][] data) {
        for (int c = 0; c < data.length; c++) {
            for (int i = 0; i < data[c].length; i++) {
                for (int j = 0; j < data[c][i].length; j++) {
                    System.out.print(data[c][i][j] + " ");
                }
                System.out.println("");
            }
            System.out.println("-------------------------");
        }
    }

    public static void printImageToOne(float[][][] data) {
        for (int c = 0; c < data.length; c++) {
            for (int i = 0; i < data[c].length; i++) {
                for (int j = 0; j < data[c][i].length; j++) {
                    if (data[c][i][j] == 0.0) {
                        System.out.print("0 ");
                    } else {
                        System.out.print("1 ");
                    }
                }
                System.out.println("");
            }
            System.out.println("-------------------------");
        }
    }

    public static void printImage(float[][][][] data) {
        for (int m = 0; m < data.length; m++) {
            for (int c = 0; c < data[m].length; c++) {
                for (int i = 0; i < data[m][c].length; i++) {
                    for (int j = 0; j < data[m][c][i].length; j++) {
                        System.out.print(data[m][c][i][j] + " ");
                    }
                    System.out.println("");
                }
                System.out.println("-------------------------");
            }
            System.out.println("=============================");
        }
    }
}

