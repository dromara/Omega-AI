package com.omega.engine.updater;

/**
 * @author Administrator
 * <p>
 * <p>
 * <p>
 * v = beta * v - learning_rate * d
 */
public class MomentumUtils {
    public static float mu = 0.9f;

    public static float[] momentum(float[] x, float[] runing, float learnRate) {
        if (runing == null) {
            runing = new float[x.length];
        }
        for (int c = 0; c < runing.length; c++) {
            runing[c] = mu * runing[c] - learnRate * x[c];
        }
        return runing;
    }

    public static float[][] momentum(float[][] x, float[][] runing, float learnRate) {
        if (runing == null) {
            runing = new float[x.length][x[0].length];
        }
        for (int c = 0; c < runing.length; c++) {
            for (int h = 0; h < runing[c].length; h++) {
                runing[c][h] = mu * runing[c][h] - learnRate * x[c][h];
            }
        }
        return runing;
    }

    public static float[][][] momentum(float[][][] x, float[][][] runing, float learnRate) {
        if (runing == null) {
            runing = new float[x.length][x[0].length][x[0][0].length];
        }
        for (int c = 0; c < runing.length; c++) {
            for (int h = 0; h < runing[c].length; h++) {
                for (int w = 0; w < runing[c][h].length; w++) {
                    runing[c][h][w] = mu * runing[c][h][w] - learnRate * x[c][h][w];
                }
            }
        }
        return runing;
    }

    public static float[][][][] momentum(float[][][][] x, float[][][][] runing, float learnRate) {
        if (runing == null) {
            runing = new float[x.length][x[0].length][x[0][0].length][x[0][0][0].length];
        }
        for (int n = 0; n < runing.length; n++) {
            for (int c = 0; c < runing[n].length; c++) {
                for (int h = 0; h < runing[n][c].length; h++) {
                    for (int w = 0; w < runing[n][c][h].length; w++) {
                        runing[n][c][h][w] = mu * runing[n][c][h][w] - learnRate * x[n][c][h][w];
                    }
                }
            }
        }
        return runing;
    }
}

